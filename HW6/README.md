# CB_HWs
This is my CB homework 6

DATA INFO:

    1. GIL:

        positive number 4247
        negative number 7837
        Length max: 37   Length min: 3

    2. NLV:

        positive number 4569
        negative number 8968
        Length max: 37   Length min: 3

    totoal number = 25621
    amino acid dict legth = 29
    amino_acid = {'-': 0, 'C': 1, 'A': 2, 'S': 3, 'I': 4, 'P': 5, 'G': 6, 'E': 7, 'F': 8, '=': 9, 'K': 10, 'N': 11, 'T': 12, 'W': 13, 'D': 14, 'R': 15, 'Y': 16, 'V': 17, 'L': 18, 'Q': 19, 'H': 20, 'M': 21, 'X': 22, 'B': 23, 'O': 24, '*': 25, '#': 26, '7': 27, '1': 28}

PERFORMANCE of GIL:

    1. trivial classification

    best valid: 0.764835
    best test: 0.751412

    2. modified version with simplified data loading:

    best valid: 0.772849
    best test: 0.757610

    3. improved version with random input

    best valid: 0.870361
    best test: 0.881962

    4. padding modified version

    best valid: 0.880625
    best test: 0.901695

PERFORMANCE of NLV:

    1. trivial classification

    best valid: 0.831547
    best test: 0.844104

    2. modified version with simplified data loading:

    best valid: 0.836550
    best test: 0.840567

    3. improved version with random input

    best valid: 0.891274
    best test: 0.906078

    4. padding modified version

    best valid: 0.907562
    best test: 0.921636