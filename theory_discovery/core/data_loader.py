"""Future streaming loaders. No real-data access without an explicit gate."""
import csv
import json
from pathlib import Path
from .schemas import EditingPair, require


def load_pairs(config):
    # Gate before opening a file, including a nonexistent path.
    require(config.allow_real_data, 'Real-data loading disabled; construction-only configuration')
    require(bool(config.input_path), 'Set input_path explicitly')
    seen = set()
    with Path(config.input_path).open(encoding='utf-8') as handle:
        rows = csv.reader(handle, delimiter='\t') if config.input_format == 'wnc_tsv' else handle
        for index, row in enumerate(rows):
            if index >= config.max_observations:
                break
            if config.input_format == 'wnc_tsv':
                require(len(row) == 7, f'Expected 7 WNC columns on line {index + 1}; verify dataset revision')
                # Adapter assumption only. No WNC file has been opened during development.
                pair = EditingPair(f'wnc-line-{index + 1}', row[3], row[4], metadata={'line_number': index + 1})
            else:
                raw = json.loads(row)
                if set(raw) == {'before', 'after'}:
                    pair = EditingPair(f'case-{index + 1:04d}', raw['before'], raw['after'])
                else:
                    pair = EditingPair(**raw)
            require(pair.split == 'discovery', 'Development/test cases cannot enter discovery')
            require(pair.case_id not in seen, 'Duplicate case ID')
            require(set(pair.pre_context) <= set(config.pre_context_keys), 'Undeclared pre-edit context')
            seen.add(pair.case_id)
            yield pair


def assert_disjoint_splits(discovery, development, test):
    sets = [set(s) for s in (discovery, development, test)]
    require(not any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)), 'Split ID leakage')
