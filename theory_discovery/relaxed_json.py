"""Recover harmless JSON presentation defects without inventing missing decisions."""
import json


def _without_trailing_commas(text):
    output=[];quoted=False;escaped=False;removed=False
    for i,char in enumerate(text):
        if quoted:
            output.append(char)
            if escaped:escaped=False
            elif char=='\\':escaped=True
            elif char=='"':quoted=False
            continue
        if char=='"':quoted=True
        if char==',':
            j=i+1
            while j<len(text) and text[j].isspace():j+=1
            if j<len(text) and text[j] in '}]':
                removed=True
                continue
        output.append(char)
    return ''.join(output),removed


def relaxed_object(content):
    if not isinstance(content,str):raise ValueError('Model content is not text')
    text=content.lstrip('\ufeff').strip();start=text.find('{')
    if start<0:raise ValueError('No JSON object')
    notes=[]
    if start:notes.append('ignored_presentation_prefix')
    text,changed=_without_trailing_commas(text[start:])
    if changed:notes.append('removed_trailing_commas')
    value,end=json.JSONDecoder().raw_decode(text)
    if not isinstance(value,dict):raise ValueError('Expected JSON object')
    suffix=text[end:].strip()
    if suffix and suffix!=chr(96)*3:
        if suffix.startswith('{'):raise ValueError('Ambiguous multiple JSON objects')
        notes.append('ignored_presentation_suffix')
    elif suffix:notes.append('removed_code_fence')
    return value,notes


def complete_prefix_fields(content):
    """Return only top-level fields whose JSON values are fully closed."""
    if not isinstance(content,str):return {}
    start=content.find('{')
    if start<0:return {}
    text,_=_without_trailing_commas(content[start:])
    decoder=json.JSONDecoder();offset=1;fields={}
    while offset<len(text):
        while offset<len(text) and text[offset].isspace():offset+=1
        if offset>=len(text) or text[offset]=='}':break
        try:
            key,end=decoder.raw_decode(text,offset)
            if not isinstance(key,str):break
            offset=end
            while offset<len(text) and text[offset].isspace():offset+=1
            if offset>=len(text) or text[offset]!=':':break
            offset+=1
            while offset<len(text) and text[offset].isspace():offset+=1
            value,offset=decoder.raw_decode(text,offset)
        except (ValueError,TypeError):break
        fields[key]=value
        while offset<len(text) and text[offset].isspace():offset+=1
        if offset>=len(text) or text[offset]!=',':break
        offset+=1
    return fields


def recover_content(content,wire_payload):
    try:return relaxed_object(content)
    except (ValueError,TypeError):
        fields=complete_prefix_fields(content)
        if 'outside_condition_profiles' in wire_payload:
            required={'similar_case_ids'}
            defaults={'reason':'The complete selected-case list was recovered; the explanatory tail was incomplete.'}
        elif 'own_support_case_ids' in wire_payload and 'matched_outside_case_ids' in wire_payload:
            required={'boundary_text','supporting_case_ids','contradicting_case_ids'}
            defaults={'rationale':'Core boundary and evidence were complete; the explanatory tail was incomplete.',
                      'unresolved_questions':[]}
        elif 'rules' in wire_payload:
            required={'branches','excluded_rules'}
            defaults={'open_questions':[]}
        elif 'theory_situations' in wire_payload:
            required={'abstain','branch_id','source_rule_ids'}
            defaults={'reason':'A complete frozen-theory choice was recovered; the explanatory tail was incomplete.'}
        else:raise ValueError('Incomplete core JSON result')
        if not required<=set(fields):raise ValueError('Incomplete core JSON result')
        for key,value in defaults.items():fields.setdefault(key,value)
        return fields,['recovered_complete_core_fields','omitted_incomplete_explanatory_tail']
