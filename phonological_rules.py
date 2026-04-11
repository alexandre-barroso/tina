import itertools
import re

def generate_variants(transcription):
    
    
    dialect_profiles = get_dialect_profiles()
    
    
    core = transcription.strip("/")
    
    
    base_variants = set([core])  
    
    
    consonant_epenthesis_variants = generate_consonant_epenthesis_variants(core)
    
    
    variants_to_process = base_variants.union(consonant_epenthesis_variants)
    cs_epenthesis_variants = generate_cs_epenthesis_variants(variants_to_process)
    
    
    all_base_variants = base_variants.union(consonant_epenthesis_variants).union(cs_epenthesis_variants)
    
    
    vowel_harmony_variants = generate_vowel_harmony_variants(all_base_variants)
    
    
    all_variants = all_base_variants.union(vowel_harmony_variants)
    
    
    unique_variants = generate_dialect_variants(all_variants, dialect_profiles)
    
    
    variants_list = sorted(unique_variants)
    variants_string = " ".join(variants_list)
    variant_count = len(variants_list)
    
    return variants_string, variant_count

def generate_vowel_harmony_variants(variants):
    
    new_variants = set()

    for variant in variants:
        expanded_variants = _expand_archiphonemic_vowel_markers(variant)

        for expanded_variant in expanded_variants:
            new_variants.add(expanded_variant)

            harmonized_variant = _apply_stress_conditioned_vowel_harmony(expanded_variant)
            if harmonized_variant is not None and harmonized_variant != expanded_variant:
                new_variants.add(harmonized_variant)

    return new_variants


def _expand_archiphonemic_vowel_markers(variant):
    
    pattern = re.compile(r'(?P<cons>[td])?(?P<marker>[IU])')
    matches = list(pattern.finditer(variant))

    if not matches:
        return {variant}

    replacement_options = []

    for match in matches:
        cons = match.group("cons")
        marker = match.group("marker")

        vowel_options = {'I': ['e', 'ɪ'], 'U': ['o', 'ʊ']}[marker]

        if cons == 't':
            cons_options = ['t', 'tʃ']
        elif cons == 'd':
            cons_options = ['d', 'dʒ']
        else:
            cons_options = ['']

        replacement_options.append(
            [(new_cons, new_vowel) for new_cons in cons_options for new_vowel in vowel_options]
        )

    expanded_variants = set()

    for combo in itertools.product(*replacement_options):
        modified_variant = variant

        for match, replacement in sorted(zip(matches, combo), key=lambda item: item[0].start(), reverse=True):
            start, end = match.span()
            new_segment = replacement[0] + replacement[1]
            modified_variant = modified_variant[:start] + new_segment + modified_variant[end:]

        expanded_variants.add(modified_variant)

    return expanded_variants


_PRECOMPOSED_NASAL_TO_BASE = {
    'ã': 'a',
    'ẽ': 'e',
    'ĩ': 'i',
    'õ': 'o',
    'ũ': 'u',
}

_BASE_TO_PRECOMPOSED_NASAL = {
    'a': 'ã',
    'e': 'ẽ',
    'i': 'ĩ',
    'o': 'õ',
    'u': 'ũ',
}

_VOWEL_UNIT_PATTERN = re.compile(
    r'ɛ̃|ɔ̃|[aeiouɛɔɪʊEOIU]\u0303|[aeiouɛɔɪʊEOIU][\u0303]?|[ãẽĩõũ]'
)

_GLIDE_OR_DIPHTHONG_CHARS = {'y', 'w', 'ỹ'}

_STRESSED_HARMONY_MAP = {
    
    
    
    
    'ɛ': {'e': 'ɛ', 'o': 'ɔ'},
    'ɔ': {'e': 'ɛ', 'o': 'ɔ'},

    
    'u': {'e': 'i', 'ɛ': 'i', 'o': 'u', 'ɔ': 'u'},
    'i': {'e': 'i', 'ɛ': 'i', 'o': 'u', 'ɔ': 'u'},
}


def _apply_stress_conditioned_vowel_harmony(variant):
    
    syllables = variant.split('.')
    stressed_indices = [i for i, syllable in enumerate(syllables) if 'ˈ' in syllable]

    if len(stressed_indices) != 1:
        return None

    stressed_index = stressed_indices[0]
    if stressed_index == 0:
        return None

    stressed_info = _extract_single_simple_vowel(syllables[stressed_index])
    preceding_info = _extract_single_simple_vowel(syllables[stressed_index - 1])

    if stressed_info is None or preceding_info is None:
        return None

    stressed_base, _ = _split_vowel_token(stressed_info["token"])
    preceding_base, preceding_nasal = _split_vowel_token(preceding_info["token"])

    harmony_targets = _STRESSED_HARMONY_MAP.get(stressed_base)
    if harmony_targets is None:
        return None

    replacement_base = harmony_targets.get(preceding_base)
    if replacement_base is None:
        return None

    replacement_token = _build_vowel_token(replacement_base, preceding_nasal)

    if replacement_token == preceding_info["token"]:
        return None

    new_preceding_syllable = (
        preceding_info["syllable"][:preceding_info["start"]]
        + replacement_token
        + preceding_info["syllable"][preceding_info["end"]:]
    )

    new_syllables = list(syllables)
    new_syllables[stressed_index - 1] = new_preceding_syllable
    return '.'.join(new_syllables)


def _extract_single_simple_vowel(syllable):
    
    bare_syllable = syllable.replace('ˈ', '')
    matches = list(_VOWEL_UNIT_PATTERN.finditer(bare_syllable))

    if len(matches) != 1:
        return None

    match = matches[0]

    prev_char = bare_syllable[match.start() - 1] if match.start() > 0 else ''
    next_char = bare_syllable[match.end()] if match.end() < len(bare_syllable) else ''

    if prev_char in _GLIDE_OR_DIPHTHONG_CHARS or next_char in _GLIDE_OR_DIPHTHONG_CHARS:
        return None

    return {
        "syllable": bare_syllable,
        "token": match.group(),
        "start": match.start(),
        "end": match.end(),
    }


def _split_vowel_token(token):
    
    if token in _PRECOMPOSED_NASAL_TO_BASE:
        return _PRECOMPOSED_NASAL_TO_BASE[token], True

    if token.endswith('\u0303'):
        return token[:-1], True

    return token, False


def _build_vowel_token(base, nasal):
    
    if not nasal:
        return base

    if base in _BASE_TO_PRECOMPOSED_NASAL:
        return _BASE_TO_PRECOMPOSED_NASAL[base]

    return base + '\u0303'


def get_dialect_profiles():
    
    return {
        "Nortista_Nordestino": {
            'H': ['ʁ', 'h'],
            'T': ['t'],
            'D': ['d'],
            'S': ['ʃ', 's'],
            'R': ['ʁ', 'h', 'χ'],
            'L': ['l'],
            'E':['ɛ'],
            'O':['ɔ']
        },
        "Sulista_Sudestino": {
            'H': ['ʁ', 'h'],
            'T': ['tʃ'],
            'D': ['dʒ'],
            'S': ['s'],
            'R': ['ɾ', 'ɻ'],
            'L': ['ʎ'],
            'E':['e'],
            'O':['o']
        },
        "Carioca": {
            'H': ['ʁ', 'h'],
            'T': ['tʃ'],
            'D': ['dʒ'],
            'S': ['s', 'ʃ'],
            'R': ['ʁ', 'χ', 'h'],
            'L': ['ʎ'],
            'E':['e'],
            'O':['o']
        }
    }

def generate_consonant_epenthesis_variants(core):
    
    all_epenthesis_variants = set()
    
    
    consonants = "bdDfgkpqtTv"
    
    
    coda_positions = find_consonant_coda_positions(core, consonants)
    
    
    if coda_positions:
        all_epenthesis_variants = create_epenthesis_variants(core, coda_positions)
    
    return all_epenthesis_variants

def find_consonant_coda_positions(core, consonants):
    
    coda_positions = []
    
    
    for match in re.finditer(f"([{consonants}])\\.", core):
        coda_positions.append((match.start(1), match.end(1), "."))
    
    
    if core and core[-1] in consonants:
        coda_positions.append((len(core)-1, len(core), "/"))
    
    return coda_positions

def create_epenthesis_variants(core, positions):
    
    epenthesis_variants = set()
    
    
    for r in range(1, len(positions) + 1):
        for positions_subset in itertools.combinations(positions, r):
            
            new_variant = list(core)
            
            
            for start, end, sep_type in sorted(positions_subset, reverse=True):
                if sep_type == ".":
                    
                    new_variant[start:end+1] = f".{core[start:end]}i."
                else:  
                    
                    new_variant.insert(end, "i")
            
            epenthesis_variants.add(''.join(new_variant))
    
    return epenthesis_variants

def generate_cs_epenthesis_variants(variants_to_process):
    
    cs_epenthesis_variants = set()
    
    
    all_consonants_for_s = "bdDfgkmnptTvszʃ"
    
    for variant in variants_to_process:
        
        onset_positions = find_cs_onset_positions(variant, all_consonants_for_s)
        coda_positions = find_cs_coda_positions(variant, all_consonants_for_s)
        
        
        all_positions = onset_positions + coda_positions
        
        
        if all_positions:
            cs_epenthesis_variants.update(
                create_cs_epenthesis_variants(variant, all_positions))
    
    return cs_epenthesis_variants

def find_cs_onset_positions(variant, all_consonants_for_s):
    
    onset_positions = []
    
    
    for match in re.finditer(f"\\.(([{all_consonants_for_s}])s)", variant):
        consonant = match.group(2)
        
        if consonant not in ['s', 'ʃ']:
            onset_positions.append((match.start(1), match.end(1), consonant, "onset"))
    
    
    if len(variant) >= 2 and variant[0] in all_consonants_for_s and variant[1] == 's':
        
        if variant[0] not in ['s', 'ʃ']:
            onset_positions.append((0, 2, variant[0], "onset"))
    
    return onset_positions

def find_cs_coda_positions(variant, all_consonants_for_s):
    
    coda_positions = []
    
    
    for match in re.finditer(f"([{all_consonants_for_s}])s\\.", variant):
        consonant = match.group(1)
        
        if consonant not in ['s', 'ʃ']:
            
            coda_positions.append((match.start(0), match.end(0), consonant, "coda"))
    
    
    if len(variant) >= 2 and variant[-2] in all_consonants_for_s and variant[-1] == 's':
        consonant = variant[-2]
        
        if consonant != 'k' and consonant not in ['s', 'ʃ']:
            coda_positions.append((len(variant)-2, len(variant), consonant, "coda"))
    
    return coda_positions

def create_cs_epenthesis_variants(variant, positions):
    
    epenthesis_variants = set()
    
    
    for r in range(1, len(positions) + 1):
        for positions_subset in itertools.combinations(positions, r):
            
            variant_list = list(variant)
            
            
            for start, end, consonant, position_type in sorted(positions_subset, reverse=True):
                if position_type == "onset":
                    
                    if start > 0 and variant_list[start-1] == '.':
                        
                        variant_list[start:end] = f"{consonant}i.s"
                    else:
                        
                        variant_list[start:end] = f"{consonant}i.s"
                else:  
                    
                    if end < len(variant_list) and variant_list[end-1] == '.':
                        
                        variant_list[start:end] = f".{consonant}is."
                    else:
                        
                        variant_list[start:end] = f".{consonant}is"
            
            epenthesis_variants.add(''.join(variant_list))
    
    return epenthesis_variants

def generate_dialect_variants(base_variants, dialect_profiles):
    
    unique_variants = set()
    
    for base_variant in base_variants:
        for mapping in dialect_profiles.values():
            
            ambiguous_letters = sorted({ch for ch in base_variant if ch in mapping})
            
            
            
            
            if ambiguous_letters:
                
                unique_variants.update(
                    create_dialect_variants(base_variant, ambiguous_letters, mapping))
            else:
                unique_variants.add(f"/{base_variant}/")
    
    return unique_variants

def create_dialect_variants(variant, ambiguous_letters, mapping):
    
    dialect_variants = set()
    
    
    has_final_r = variant.endswith('R')
    
    
    rs_positions = []
    for match in re.finditer('RS', variant):
        rs_positions.append(match.start())
    
    
    
    replacements = {}
    for letter in ambiguous_letters:
        if letter == 'R' and has_final_r:
            
            
            replacements[letter] = mapping[letter] + ['']
        else:
            replacements[letter] = mapping[letter]
    
    
    letter_replacements = []
    for letter in ambiguous_letters:
        letter_replacements.append([(letter, repl) for repl in replacements[letter]])
    
    
    for combo in itertools.product(*letter_replacements):
        
        replacement_dict = {}
        for letter, repl in combo:
            replacement_dict[letter] = repl
        
        
        new_variant = list(variant)
        
        
        for i, char in enumerate(variant):
            if char in replacement_dict:
                
                if char == 'R' and i == len(variant) - 1 and replacement_dict[char] == '':
                    new_variant[i] = ''
                
                elif char == 'R' and i in rs_positions:
                    if replacement_dict[char] in ['ɾ', 'ɻ']:
                        new_variant[i] = replacement_dict[char]
                    else:
                        new_variant[i] = ''  
                
                else:
                    
                    replacement = replacement_dict[char]
                    if char == 'R' and replacement == '':
                        
                        
                        for possible_r in mapping['R']:
                            if possible_r:  
                                replacement = possible_r
                                break
                    new_variant[i] = replacement
        
        
        final_variant = ''.join(new_variant)
        
        final_variant = re.sub(r'\.\.', '.', final_variant)
        
        final_variant = final_variant.strip('.')
        
        dialect_variants.add(f"/{final_variant}/")
    
    return dialect_variants
