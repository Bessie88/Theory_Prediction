"""Request-local short labels for structured hash references; source text stays literal."""
import re

_HASH_ID = re.compile(r'^[a-z][a-z_]*_[0-9a-f]{20}$')
_LITERAL_FIELDS = {'before', 'after', 'before_text', 'after_text', 'quote', 'target_expression',
                   'name', 'definition', 'rule_text', 'natural_rule_text', 'rationale', 'reason',
                   'interpretation', 'situation', 'transformation', 'method', 'target_feature',
                   'discovery_note', 'comparison_memo'}
_PREFIXES = {'condition':'C', 'group':'G', 'operation':'O', 'rule':'R', 'natural_rule':'N',
             'hypothesis':'H', 'branch':'B', 'action':'A', 'boundary':'D'}

def short_reference_payload(payload):
    """Return wire payload and short->original map; never alter caller-owned data."""
    ids=set();occupied=set()
    def collect(value,literal=False):
        if isinstance(value,str):
            occupied.add(value)
            if not literal and _HASH_ID.fullmatch(value):ids.add(value)
        elif isinstance(value,list):
            for item in value:collect(item,literal)
        elif isinstance(value,dict):
            for key,item in value.items():
                occupied.add(key)
                if not literal and _HASH_ID.fullmatch(key):ids.add(key)
                collect(item,literal or key in _LITERAL_FIELDS)
    collect(payload)
    forward={};counts={}
    for reference in sorted(ids):
        family=reference.rsplit('_',1)[0];prefix=_PREFIXES.get(family,'X')
        count=counts.get(prefix,0)+1
        while prefix+str(count) in occupied:count+=1
        alias=prefix+str(count);counts[prefix]=count
        occupied.add(alias);forward[reference]=alias
    def encode(value,literal=False):
        if isinstance(value,str):return value if literal else forward.get(value,value)
        if isinstance(value,list):return [encode(x,literal) for x in value]
        if isinstance(value,dict):
            return {(key if literal else forward.get(key,key)):encode(item,literal or key in _LITERAL_FIELDS)
                    for key,item in value.items()}
        return value
    return encode(payload),{alias:reference for reference,alias in forward.items()}

def restore_reference_ids(value,reverse):
    if isinstance(value,str):return reverse.get(value,value)
    if isinstance(value,list):return [restore_reference_ids(x,reverse) for x in value]
    if isinstance(value,dict):
        return {reverse.get(key,key):restore_reference_ids(item,reverse) for key,item in value.items()}
    return value
