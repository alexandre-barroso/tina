import itertools
import re

def generate_variants(transcription):
    """
    Main function that coordinates the generation of all phonetic variants.
    
    Args:
        transcription (str): A transcription string with arch phonemes (e.g. '/maw.ko.ˈHẽ.Tɪ/')
    
    Returns:
        tuple: A tuple containing:
           - str: A string of unique variant transcriptions, e.g. "/variant1/ /variant2/ ..."
           - int: The number of valid variants generated.
    """
    # Define dialect profiles
    dialect_profiles = get_dialect_profiles()
    
    # Strip slashes from the input
    core = transcription.strip("/")
    
    # Generate base variants with epenthesis
    base_variants = set([core])  # Start with the original transcription
    
    # Generate consonant coda epenthesis variants
    consonant_epenthesis_variants = generate_consonant_epenthesis_variants(core)
    
    # Generate consonant+s epenthesis variants
    variants_to_process = base_variants.union(consonant_epenthesis_variants)
    cs_epenthesis_variants = generate_cs_epenthesis_variants(variants_to_process)
    
    # Combine all base variants with all epenthesis variants
    all_base_variants = base_variants.union(consonant_epenthesis_variants).union(cs_epenthesis_variants)
    
    # Add vowel harmony variants
    vowel_harmony_variants = generate_vowel_harmony_variants(all_base_variants)
    
    # Combine with previous variants
    all_variants = all_base_variants.union(vowel_harmony_variants)
    
    # Generate dialect-specific variants
    unique_variants = generate_dialect_variants(all_variants, dialect_profiles)
    
    # Format and return results
    variants_list = sorted(unique_variants)
    variants_string = " ".join(variants_list)
    variant_count = len(variants_list)
    
    return variants_string, variant_count

def generate_vowel_harmony_variants(variants):
    """
    Generate phonetic variants based on two mechanisms:

    1) Archiphonemic vowel expansion:
       - 'I' -> 'e' or 'ɪ'
       - 'U' -> 'o' or 'ʊ'
       - If preceded by 't' or 'd', also allow:
         - 't' -> 't' or 'tʃ'
         - 'd' -> 'd' or 'dʒ'

    2) Stress-conditioned vowel harmony:
       - If the stressed syllable contains only a single eligible vowel nucleus,
         and the immediately preceding syllable also contains only a single
         eligible vowel nucleus, generate an additional harmonized variant.

       Harmony rules implemented:
         - stressed /ɛ, ɛ̃/  <- preceding /e, ẽ/  becomes /ɛ, ɛ̃/
         - stressed /ɔ, ɔ̃/  <- preceding /o, õ/  becomes /ɔ, ɔ̃/
         - stressed /u/      <- preceding /e, ẽ, ɛ, ɛ̃/ becomes /i, ĩ/
                               preceding /o, õ, ɔ, ɔ̃/ becomes /u, ũ/
         - stressed /i/      <- preceding /e, ẽ, ɛ, ɛ̃/ becomes /i, ĩ/
                               preceding /o, õ, ɔ, ɔ̃/ becomes /u, ũ/

       The harmony applies only when both syllables have a simple nucleus:
       no diphthongs or other complex nuclei such as 'ey', 'ew', 'ẽỹ', etc.

    Args:
        variants (set): A set of transcription strings.

    Returns:
        set: A set of variants with archiphonemic expansion and optional
             stress-conditioned vowel harmony.
    """
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
    """
    Expand archiphonemic vowel markers I/U, preserving the original behavior
    of the rule layer.
    """
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
    # Open-mid stressed vowels trigger lowering in the immediately preceding
    # syllable by openness class, not by matching front/back quality:
    #   e/ẽ -> ɛ/ɛ̃
    #   o/õ -> ɔ/ɔ̃
    'ɛ': {'e': 'ɛ', 'o': 'ɔ'},
    'ɔ': {'e': 'ɛ', 'o': 'ɔ'},

    # High-vowel harmony / raising
    'u': {'e': 'i', 'ɛ': 'i', 'o': 'u', 'ɔ': 'u'},
    'i': {'e': 'i', 'ɛ': 'i', 'o': 'u', 'ɔ': 'u'},
}


def _apply_stress_conditioned_vowel_harmony(variant):
    """
    Apply the requested harmony only across the stressed syllable and the
    immediately preceding syllable.
    """
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
    """
    Return information about the syllable's only vowel if it is a simple
    non-diphthongal nucleus. Otherwise return None.
    """
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
    """
    Split a vowel token into (base_vowel, is_nasal).
    """
    if token in _PRECOMPOSED_NASAL_TO_BASE:
        return _PRECOMPOSED_NASAL_TO_BASE[token], True

    if token.endswith('\u0303'):
        return token[:-1], True

    return token, False


def _build_vowel_token(base, nasal):
    """
    Build a vowel token, preserving nasalization when requested.
    """
    if not nasal:
        return base

    if base in _BASE_TO_PRECOMPOSED_NASAL:
        return _BASE_TO_PRECOMPOSED_NASAL[base]

    return base + '\u0303'


def get_dialect_profiles():
    """
    Define and return dialect profiles for Brazilian Portuguese variants.
    
    Returns:
        dict: A dictionary mapping dialect names to their phoneme replacements.
    """
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
    """
    Generate variants with epenthetic vowel 'i' after consonants in coda position.
    
    Args:
        core (str): The core transcription without slashes.
        
    Returns:
        set: Set of variants with consonant coda epenthesis.
    """
    all_epenthesis_variants = set()
    
    # Define consonants that can trigger epenthesis in coda position
    consonants = "bdDfgkpqtTv"
    
    # Find positions of consonants in coda
    coda_positions = find_consonant_coda_positions(core, consonants)
    
    # If we found any coda consonants, create epenthesis variants
    if coda_positions:
        all_epenthesis_variants = create_epenthesis_variants(core, coda_positions)
    
    return all_epenthesis_variants

def find_consonant_coda_positions(core, consonants):
    """
    Find positions of consonants in coda position.
    
    Args:
        core (str): The core transcription without slashes.
        consonants (str): String of consonant symbols to match.
        
    Returns:
        list: List of tuples (start, end, separator) for each coda consonant position.
    """
    coda_positions = []
    
    # Search for consonant + . pattern
    for match in re.finditer(f"([{consonants}])\\.", core):
        coda_positions.append((match.start(1), match.end(1), "."))
    
    # Search for consonant at the very end (before /)
    if core and core[-1] in consonants:
        coda_positions.append((len(core)-1, len(core), "/"))
    
    return coda_positions

def create_epenthesis_variants(core, positions):
    """
    Create variants with epenthesis at specified positions.
    
    Args:
        core (str): The core transcription without slashes.
        positions (list): List of position tuples (start, end, separator).
        
    Returns:
        set: Set of variants with epenthesis applied.
    """
    epenthesis_variants = set()
    
    # Process all combinations of epenthesis positions
    for r in range(1, len(positions) + 1):
        for positions_subset in itertools.combinations(positions, r):
            # Create a new variant with epenthesis at selected positions
            new_variant = list(core)
            
            # Apply epenthesis from right to left to avoid affecting indices
            for start, end, sep_type in sorted(positions_subset, reverse=True):
                if sep_type == ".":
                    # Replace C. with .Ci.
                    new_variant[start:end+1] = f".{core[start:end]}i."
                else:  # sep_type == "/"
                    # Add i before final /
                    new_variant.insert(end, "i")
            
            epenthesis_variants.add(''.join(new_variant))
    
    return epenthesis_variants

def generate_cs_epenthesis_variants(variants_to_process):
    """
    Generate variants with epenthesis for consonant+s clusters.
    
    Args:
        variants_to_process (set): Set of base variants to process.
        
    Returns:
        set: Set of variants with consonant+s epenthesis.
    """
    cs_epenthesis_variants = set()
    
    # Include all consonants that might appear before 's'
    all_consonants_for_s = "bdDfgkmnptTvszʃ"
    
    for variant in variants_to_process:
        # Find all Cs positions (onset and coda)
        onset_positions = find_cs_onset_positions(variant, all_consonants_for_s)
        coda_positions = find_cs_coda_positions(variant, all_consonants_for_s)
        
        # Combine all positions
        all_positions = onset_positions + coda_positions
        
        # If we found any Cs patterns, create epenthesis variants
        if all_positions:
            cs_epenthesis_variants.update(
                create_cs_epenthesis_variants(variant, all_positions))
    
    return cs_epenthesis_variants

def find_cs_onset_positions(variant, all_consonants_for_s):
    """
    Find positions of consonant+s clusters at syllable onsets.
    
    Args:
        variant (str): The variant to process.
        all_consonants_for_s (str): String of consonant symbols to match.
        
    Returns:
        list: List of position tuples for onset positions.
    """
    onset_positions = []
    
    # Look for .Cs pattern (consonant + s at onset, after a period)
    for match in re.finditer(f"\\.(([{all_consonants_for_s}])s)", variant):
        consonant = match.group(2)
        # Only apply if it's not 's' or 'ʃ' (avoid ss or ʃs clusters)
        if consonant not in ['s', 'ʃ']:
            onset_positions.append((match.start(1), match.end(1), consonant, "onset"))
    
    # Look for /Cs pattern (consonant + s at the very beginning)
    if len(variant) >= 2 and variant[0] in all_consonants_for_s and variant[1] == 's':
        # Include 'k' for onset positions, but not 's' or 'ʃ'
        if variant[0] not in ['s', 'ʃ']:
            onset_positions.append((0, 2, variant[0], "onset"))
    
    return onset_positions

def find_cs_coda_positions(variant, all_consonants_for_s):
    """
    Find positions of consonant+s clusters at syllable codas.
    
    Args:
        variant (str): The variant to process.
        all_consonants_for_s (str): String of consonant symbols to match.
        
    Returns:
        list: List of position tuples for coda positions.
    """
    coda_positions = []
    
    # Look for consonant followed by 's' followed by period
    for match in re.finditer(f"([{all_consonants_for_s}])s\\.", variant):
        consonant = match.group(1)
        # Skip if consonant is 's' or 'ʃ' (avoid ss or ʃs clusters)
        if consonant not in ['s', 'ʃ']:
            # Use start(0) and end(0) to get the entire match
            coda_positions.append((match.start(0), match.end(0), consonant, "coda"))
    
    # Look for Cs/ pattern (consonant + s at the very end)
    if len(variant) >= 2 and variant[-2] in all_consonants_for_s and variant[-1] == 's':
        consonant = variant[-2]
        # Only exclude 'ks' at word-final position, and skip 's' or 'ʃ'
        if consonant != 'k' and consonant not in ['s', 'ʃ']:
            coda_positions.append((len(variant)-2, len(variant), consonant, "coda"))
    
    return coda_positions

def create_cs_epenthesis_variants(variant, positions):
    """
    Create variants with epenthesis for consonant+s clusters.
    
    Args:
        variant (str): The variant to process.
        positions (list): List of position tuples.
        
    Returns:
        set: Set of variants with epenthesis applied.
    """
    epenthesis_variants = set()
    
    # Process all combinations of epenthesis positions
    for r in range(1, len(positions) + 1):
        for positions_subset in itertools.combinations(positions, r):
            # Create a new variant with epenthesis at selected positions
            variant_list = list(variant)
            
            # Apply epenthesis from right to left to avoid affecting indices
            for start, end, consonant, position_type in sorted(positions_subset, reverse=True):
                if position_type == "onset":
                    # For onset (.Cs or /Cs), replace with Ci.s
                    if start > 0 and variant_list[start-1] == '.':
                        # .Cs -> .Ci.s
                        variant_list[start:end] = f"{consonant}i.s"
                    else:
                        # /Cs -> /Ci.s (start of string)
                        variant_list[start:end] = f"{consonant}i.s"
                else:  # position_type == "coda"
                    # For coda (Cs. or Cs/), replace with .Cis.
                    if end < len(variant_list) and variant_list[end-1] == '.':
                        # Cs. -> .Cis.
                        variant_list[start:end] = f".{consonant}is."
                    else:
                        # Cs/ -> .Cis/ (end of string)
                        variant_list[start:end] = f".{consonant}is"
            
            epenthesis_variants.add(''.join(variant_list))
    
    return epenthesis_variants

def generate_dialect_variants(base_variants, dialect_profiles):
    """
    Generate all possible dialect variants from the base variants.
    
    Args:
        base_variants (set): Set of base variants to process.
        dialect_profiles (dict): Dictionary of dialect profiles.
        
    Returns:
        set: Set of all dialect variants.
    """
    unique_variants = set()
    
    for base_variant in base_variants:
        for mapping in dialect_profiles.values():
            # Find all ambiguous letters in this variant
            ambiguous_letters = sorted({ch for ch in base_variant if ch in mapping})
            
            # Note: We no longer modify the mapping here for final R
            # as that's now handled directly in create_dialect_variants
            
            if ambiguous_letters:
                # Generate all possible combinations of replacements
                unique_variants.update(
                    create_dialect_variants(base_variant, ambiguous_letters, mapping))
            else:
                unique_variants.add(f"/{base_variant}/")
    
    return unique_variants

def create_dialect_variants(variant, ambiguous_letters, mapping):
    """
    Create all possible dialect variants for a given variant,
    ensuring consistent replacements of the same phoneme within a word,
    with special exceptions for:
    1. Word-final 'R' which may be dropped (only at the very end of the word)
    2. 'R' in 'RS' clusters which is dropped except when realized as 'ɾ' or 'ɻ'
    
    Args:
        variant (str): The variant to process.
        ambiguous_letters (list): List of ambiguous letters in the variant.
        mapping (dict): The dialect mapping.
        
    Returns:
        set: Set of dialect variants.
    """
    dialect_variants = set()
    
    # Special handling for 'R': check if the last character is 'R'
    has_final_r = variant.endswith('R')
    
    # Find 'RS' clusters
    rs_positions = []
    for match in re.finditer('RS', variant):
        rs_positions.append(match.start())
    
    # Create replacement combinations for each ambiguous letter
    # Each letter will be replaced consistently throughout the word
    replacements = {}
    for letter in ambiguous_letters:
        if letter == 'R' and has_final_r:
            # For 'R', if there's a final R, we include an empty string as a possible replacement,
            # but only for the final position
            replacements[letter] = mapping[letter] + ['']
        else:
            replacements[letter] = mapping[letter]
    
    # Generate all possible combinations of letter replacements
    letter_replacements = []
    for letter in ambiguous_letters:
        letter_replacements.append([(letter, repl) for repl in replacements[letter]])
    
    # For each combination of letter replacements
    for combo in itertools.product(*letter_replacements):
        # Create a replacement dictionary
        replacement_dict = {}
        for letter, repl in combo:
            replacement_dict[letter] = repl
        
        # Create a new variant with consistent replacements
        new_variant = list(variant)
        
        # Apply replacements for each letter position
        for i, char in enumerate(variant):
            if char in replacement_dict:
                # Special case for ONLY word-final R: we might want to replace it with an empty string
                if char == 'R' and i == len(variant) - 1 and replacement_dict[char] == '':
                    new_variant[i] = ''
                # Special case for R in RS clusters: drop R unless it's 'ɾ' or 'ɻ'
                elif char == 'R' and i in rs_positions:
                    if replacement_dict[char] in ['ɾ', 'ɻ']:
                        new_variant[i] = replacement_dict[char]
                    else:
                        new_variant[i] = ''  # Drop the R
                # All other Rs must be replaced with a valid phonetic variant, never dropped
                else:
                    # Ensure we never apply an empty string replacement for non-final Rs
                    replacement = replacement_dict[char]
                    if char == 'R' and replacement == '':
                        # If we somehow got an empty replacement for a non-final R,
                        # use the first non-empty replacement from the mapping instead
                        for possible_r in mapping['R']:
                            if possible_r:  # Not empty
                                replacement = possible_r
                                break
                    new_variant[i] = replacement
        
        # Remove any doubled empty strings that might result from RS handling
        final_variant = ''.join(new_variant)
        # Clean up any resulting double periods (from dropped phonemes)
        final_variant = re.sub(r'\.\.', '.', final_variant)
        # Clean up any leading or trailing periods
        final_variant = final_variant.strip('.')
        
        dialect_variants.add(f"/{final_variant}/")
    
    return dialect_variants
