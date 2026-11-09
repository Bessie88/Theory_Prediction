"""Budget the exact short-ID wire request and keep repair feedback bounded."""
from dataclasses import replace
import json
import os

from .llm import PLAIN_LANGUAGE_INSTRUCTION
from .reference_transport import short_reference_payload,restore_reference_ids
from .response_recovery import RecoverableQwenJSONClient
from .schemas import stable_id
from .storage import write_json

REPAIR_TOKEN_RESERVE=1024
REPAIR_INSTRUCTION=(
    'Fix the reported schema or source-accounting error using the supplied inputs. '
    'Keep supported content and source IDs. Return the complete required response.')


class WireBudgetQwenClient(RecoverableQwenJSONClient):
    """Same model payload/short labels; match context checks to actual sent content."""
    def __init__(self,config,cache,endpoint=None):
        self._declared_config=dict(vars(config))
        endpoint=endpoint or os.environ.get('SIMULATION_MODEL_BASE_URL')
        super().__init__(replace(config,base_url=endpoint) if endpoint else config,cache)

    @property
    def fingerprint(self):
        return {**super().fingerprint,'config':dict(self._declared_config)}

    def prompt_fits(self,task,payload,*,repair_reserve=False):
        wire,_=short_reference_payload(payload)
        if PLAIN_LANGUAGE_INSTRUCTION not in task:
            task+='\n'+PLAIN_LANGUAGE_INSTRUCTION
        messages=[{'role':'system','content':task},
                  {'role':'user','content':json.dumps(wire,ensure_ascii=False)}]
        tokens=self.prompt_tokens(messages)
        reserve=self.config.max_tokens+self.config.context_safety_margin
        if repair_reserve:reserve+=REPAIR_TOKEN_RESERVE
        # Each actual call is checked again, including any bounded feedback.
        return ((tokens is None or tokens+reserve<=self.config.context_length)
                and sum(len(m['content']) for m in messages)
                    +(REPAIR_TOKEN_RESERVE*4 if repair_reserve else 0)<=self.config.max_prompt_chars)

    def infer(self,task,payload):
        wire,reverse=short_reference_payload(payload)
        # The inherited adapter sees exactly what it sends, so both its hard token
        # check and character check operate on the wire rather than long hash IDs.
        # Its second short-ID pass is inert; outer restoration recovers real IDs.
        result=super().infer(task,wire)
        return restore_reference_ids(result,reverse)

    def infer_checked(self,task,payload,decode):
        current=payload
        for attempt in range(self.config.retries+1):
            raw=self.infer(task,current)
            try:
                return decode(raw)
            except (ValueError,TypeError,KeyError,IndexError) as exc:
                key=stable_id('invalid',{'task':task,'payload':current,'response':raw})
                # Preserve the FULL prior response on disk, never silently discard evidence.
                write_json(self.cache.directory/'unresolved'/(key+'.json'),
                           {'task':task,'payload':current,'response':raw,
                            'error':str(exc),'attempt':attempt+1,
                            'repair_policy':'bounded_error_feedback_full_response_preserved_on_disk'})
                if attempt==self.config.retries:
                    raise ValueError('Structured response failed validation: '+str(exc)) from exc
                current=dict(payload,validation_feedback={
                    'error':str(exc)[:500],'repair_attempt':attempt+1,
                    'instruction':REPAIR_INSTRUCTION})
        raise AssertionError('Unreachable')
