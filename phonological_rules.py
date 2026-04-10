import itertools
import re

def generate_variants(transcription):

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

    new_variants = set()
    # Pattern matches an optional 't' or 'd' (captured in group 'cons') followed by 'I' or 'U' (in group 'marker')
    pattern = re.compile(r'(?P<cons>[td])?(?P<marker>[IU])')
    
    for variant in variants:
        matches = list(pattern.finditer(variant))
        if not matches:
            new_variants.add(variant)
        else:
            # For each matched occurrence, determine the replacement options as tuples:
            # (original_cons, marker, new_cons, new_vowel)
            replacement_options = []
            for m in matches:
                cons = m.group("cons")
                marker = m.group("marker")
                # Vowel replacement options:
                vowel_options = {'I': ['e', 'ɪ'], 'U': ['o', 'ʊ']}[marker]
                # Determine consonant alternatives if the marker is preceded by 't' or 'd'
                if cons:
                    if cons == 't':
                        cons_options = ['t', 'tʃ']
                    elif cons == 'd':
                        cons_options = ['d', 'dʒ']
                else:
                    cons_options = ['']  # No preceding consonant to alter
                # Each match leads to a list of possible replacements
                replacement_options.append(
                    [(cons, marker, new_cons, new_vowel) for new_cons in cons_options for new_vowel in vowel_options]
                )
            
            # Build all combinations for this variant
            for combo in itertools.product(*replacement_options):
                # Start with the original variant string; we will replace from right to left so the indices remain valid.
                modified_variant = variant
                # Zip each found match with its chosen replacement from the combo
                for m, rep in sorted(zip(matches, combo), key=lambda x: x[0].start(), reverse=True):
                    start, end = m.span()
                    # Create the replacement segment (substituting the consonant and vowel)
                    new_segment = rep[2] + rep[3]
                    # Replace the substring within the variant
                    modified_variant = modified_variant[:start] + new_segment + modified_variant[end:]
                new_variants.add(modified_variant)
    return new_variants


def get_dialect_profiles():

    return {
        "Nortista_Nordestino": {
            'H': ['ʁ', 'h'],
            'T': ['t'],
            'D': ['d'],
            'S': ['ʃ', 's'],
            'R': ['ʁ', 'h', 'χ'],
            'L': ['ʎ'],
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
    
    # Define consonants that can trigger epenthesis in coda position
    consonants = "bdDfgkpqtTv"
    
    # Find positions of consonants in coda
    coda_positions = find_consonant_coda_positions(core, consonants)
    
    # If we found any coda consonants, create epenthesis variants
    if coda_positions:
        all_epenthesis_variants = create_epenthesis_variants(core, coda_positions)
    
    return all_epenthesis_variants

def find_consonant_coda_positions(core, consonants):

    coda_positions = []
    
    # Search for consonant + . pattern
    for match in re.finditer(f"([{consonants}])\\.", core):
        coda_positions.append((match.start(1), match.end(1), "."))
    
    # Search for consonant at the very end (before /)
    if core and core[-1] in consonants:
        coda_positions.append((len(core)-1, len(core), "/"))
    
    return coda_positions

def create_epenthesis_variants(core, positions):

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
