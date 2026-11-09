"""Self-contained future reviewer UI. No network, tracking, model calls or auto-submission."""
import json


def response_template(packet):
    # Intentionally empty: a template is not a completed response or a human label.
    return json.dumps({"packet_id": packet.packet_id, "responses": []}, ensure_ascii=False, indent=2)


def render_review_html(packet):
    data = json.dumps(packet.to_dict(), ensure_ascii=True).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    return HTML_PREFIX + data + HTML_SUFFIX


HTML_PREFIX = r"""<!doctype html><html lang="en"><meta charset="utf-8">
<title>Independent content review</title>
<style>
body{font:16px system-ui;max-width:1050px;margin:2rem auto;padding:0 1rem;background:#fafafa;color:#17212b}
section{background:white;border:1px solid #ccd2d8;padding:1.2rem;margin:1rem 0}
pre{white-space:pre-wrap;overflow-wrap:anywhere;border:1px solid #ddd;padding:.7rem}
label{display:block;margin:.5rem 0}textarea{display:block;width:96%;min-height:3rem}
button,select,input{font:inherit;margin:.25rem;padding:.3rem}
del{background:#ffe0e0}ins{background:#dcf6df;text-decoration:none}
small{color:#475467}.operation{border:1px dashed #aaa;padding:.7rem;margin:.7rem 0}
</style><h1>Independent content review</h1>
<p>Review the content and frozen definitions. Select exact source text and use an evidence button.
For an insertion or deletion, a cursor position supplies an empty-side anchor.
Review decisions are saved only when you click Download responses. No submission is sent online.</p>
<label>Your reviewer ID <input id="reviewer"></label>
<main id="tasks"></main><button id="download">Download completed responses</button>
<p id="message" role="status"></p>
<script type="application/json" id="packet">"""
HTML_SUFFIX = r"""</script><script>
"use strict";
const packet=JSON.parse(document.getElementById("packet").textContent), forms=[];
const el=(tag,text,parent)=>{const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(parent)parent.append(n);return n};
const input=(title,parent)=>{const lab=el("label",title,parent),v=el("input",undefined,lab);return v};
const area=(title,parent)=>{const lab=el("label",title,parent);return el("textarea",undefined,lab)};
const select=(title,values,parent)=>{const lab=el("label",title,parent),v=el("select",undefined,lab);
for(const value of values){const o=el("option",value||"Choose",v);o.value=value}return v};
const lines=v=>v.split("\n").map(x=>x.trim()).filter(Boolean);
function evidenceButton(title,pre,side,spans,parent){
 const b=el("button",title,parent);b.type="button";b.addEventListener("mousedown",e=>e.preventDefault());
 b.onclick=()=>{const selection=getSelection();if(!selection.rangeCount)return;const r=selection.getRangeAt(0);
 if(!pre.contains(r.startContainer)||!pre.contains(r.endContainer)){alert("Select text or a cursor in the corresponding source text.");return}
 const prefix=document.createRange();prefix.selectNodeContents(pre);prefix.setEnd(r.startContainer,r.startOffset);
 const start=Array.from(prefix.toString()).length, quote=r.toString(),end=start+Array.from(quote).length;
 spans.push({side,start,end,quote});display.textContent=JSON.stringify(spans,null,2)};
 const display=el("pre","No evidence selected.",parent);
 const clear=el("button","Clear selected evidence",parent);clear.type="button";clear.onclick=()=>{spans.length=0;display.textContent="No evidence selected."};
}
function highlighted(item,side,parent){
 const pre=el("pre",undefined,parent);
 if(!item.actual_diff){pre.textContent=side==="before"?item.before_text:(item.after_text??"After unavailable");return}
 for(const h of item.actual_diff.character_hunks){
 const span=h[side],kind=h.operation;
 if(kind==="EQUAL")pre.append(document.createTextNode(span.quote));
 else el(side==="before"?"del":"ins",span.quote||"│",pre);
 }
}
for(const item of packet.items){
 const box=el("section",undefined,document.getElementById("tasks"));
 el("h2","Case "+item.identity.case_id,box);el("small","Editor "+item.identity.editor.editor_id+" · "+item.mode,box);
 if(item.limitations.length)el("p",item.limitations.join(" "),box);
 el("h3","Before",box);const before=el("pre",item.before_text,box);
 let after=null;
 if(packet.kind==="ACTION"){
 el("h3","After",box);after=el("pre",item.after_text??"After unavailable",box);
 el("h3","Actual textual changes",box);highlighted(item,"before",box);highlighted(item,"after",box);
 el("details",undefined,box).append(el("summary","Frozen Action guidelines"),el("pre",JSON.stringify(packet.protocol,null,2)));
 } else {
 el("h3","Feature definition and extraction method",box);
 el("pre",JSON.stringify(item.feature_definition,null,2),box);
 }
 if(item.context.length){el("h3","Available source context",box);el("pre",JSON.stringify(item.context,null,2),box)}
 const shown=packet.kind==="ACTION"?item.action_label:item.feature_label;
 if(shown!==null){el("h3","Model label and supporting evidence",box);el("pre",JSON.stringify(shown,null,2),box)}
 const choices=item.mode==="BLIND_ANNOTATION"?["","ASSIGN","UNCERTAIN"]:["","ACCEPT","CORRECT","UNCERTAIN"];
 const decision=select("Review decision",choices,box),notes=area("Notes",box);
 const state={item,decision,notes,ops:[]};
 if(packet.kind==="ACTION"){
 state.status=select("Assigned annotation status",["","ANNOTATED","OTHER","UNCERTAIN","NO_CHANGE"],box);
 state.explanation=area("Annotation explanation",box);
 state.confidence=select("Confidence",["","HIGH","MEDIUM","LOW","UNCERTAIN"],box);
 state.uncertainty=area("Uncertainty reasons (one per line)",box);
 state.unassigned=[];
 el("p","Mark any changes that remain unexplained:",box);
 for(const [i,h] of (item.actual_diff?.character_hunks??[]).filter(h=>h.operation!=="EQUAL").entries()){
 const lab=el("label","Change "+(i+1)+" · "+h.operation,box),check=el("input",undefined,lab);check.type="checkbox";
 state.unassigned.push({id:h.hunk_id,check});
 }
 const add=el("button","Add observed operation",box);add.type="button";
 add.onclick=()=>{
 const row=el("div",undefined,box);row.className="operation";const op={before:[],after:[],hunks:[]};
 op.label=select("Action label",["",...packet.protocol.actions.map(a=>a.action_id)],row);
 op.explanation=area("Operation explanation",row);op.confidence=select("Operation confidence",["","HIGH","MEDIUM","LOW","UNCERTAIN"],row);
 op.uncertainty=area("Operation uncertainty (one per line)",row);
 for(const [i,h] of (item.actual_diff?.character_hunks??[]).filter(h=>h.operation!=="EQUAL").entries()){
 const lab=el("label","Supports change "+(i+1),row),check=el("input",undefined,lab);check.type="checkbox";op.hunks.push({id:h.hunk_id,check});
 }
 evidenceButton("Use Before selection",before,"before",op.before,row);
 if(after)evidenceButton("Use After selection",after,"after",op.after,row);
 const remove=el("button","Remove operation",row);remove.type="button";remove.onclick=()=>{state.ops=state.ops.filter(o=>o!==op);row.remove()};
 state.ops.push(op);
 };
 } else {
 state.value=area("Assigned feature value (JSON: true, false, number, quoted text, or structured value)",box);
 state.status=select("Assigned feature status",["","KNOWN","UNKNOWN","ERROR"],box);
 state.scope=select("Evidence scope",["","SPANS","FULL_BEFORE","METADATA","CANDIDATES"],box);
 state.metadata=input("Context evidence field names (comma separated)",box);
 state.explanation=area("Explanation",box);state.uncertainty=area("Uncertainty reasons (one per line)",box);state.evidence=[];
 evidenceButton("Use Before selection",before,"before",state.evidence,box);
 }
 forms.push(state);
}
document.getElementById("download").onclick=()=>{
 try {
 const reviewer=document.getElementById("reviewer").value.trim();if(!reviewer)throw Error("Enter your reviewer ID.");
 const responses=[];
 for(const s of forms){if(!s.decision.value)continue;let action_annotation=null,feature_annotation=null;
 if(s.decision.value==="ASSIGN"||s.decision.value==="CORRECT"){
 if(packet.kind==="ACTION"){
 action_annotation={status:s.status.value,operations:s.ops.map(o=>({
 action_id:o.label.value,diff_hunk_ids:o.hunks.filter(h=>h.check.checked).map(h=>h.id),
 before_evidence:o.before,after_evidence:o.after,explanation:o.explanation.value,
 confidence:o.confidence.value,uncertainty:lines(o.uncertainty.value)})),
 unassigned_diff_hunk_ids:s.unassigned.filter(h=>h.check.checked).map(h=>h.id),explanation:s.explanation.value,
 confidence:s.confidence.value,uncertainty:lines(s.uncertainty.value)};
 }else{
 feature_annotation={value:s.status.value==="KNOWN"?JSON.parse(s.value.value):null,status:s.status.value,
 evidence:s.evidence,context_evidence_names:s.metadata.value.split(",").map(x=>x.trim()).filter(Boolean),
 evidence_scope:s.scope.value,explanation:s.explanation.value,uncertainty:lines(s.uncertainty.value)};
 }}
 responses.push({task_id:s.item.task_id,reviewer_id:reviewer,completed_at_utc:new Date().toISOString(),
 mode:s.item.mode,decision:s.decision.value,action_annotation,feature_annotation,notes:s.notes.value});
 }
 if(!responses.length)throw Error("Complete at least one review decision.");
 const blob=new Blob([JSON.stringify({packet_id:packet.packet_id,responses},null,2)],{type:"application/json"});
 const url=URL.createObjectURL(blob),a=document.createElement("a");a.href=url;a.download="review-responses.json";a.click();
 setTimeout(()=>URL.revokeObjectURL(url),1000);
 document.getElementById("message").textContent="Responses downloaded locally. Import validation is still required.";
 }catch(error){document.getElementById("message").textContent=error.message}
};
</script></html>"""
