def calculate_average_precision(generated_caption, reference_captions):
    """
    Calculate the Average Precision (AP) for a generated caption against reference captions.
    """
    generated_tokens = generated_caption.lower().split()
    reference_tokens = set(" ".join(reference_captions).lower().split())

    # Track relevant tokens and their positions
    relevant_positions = [i for i, token in enumerate(generated_tokens) if token in reference_tokens]

    if not relevant_positions:
        return 0  # No relevant tokens, AP is 0

    # Calculate precision at each relevant position
    precisions = [(i + 1) / (pos + 1) for i, pos in enumerate(relevant_positions)]
    average_precision = sum(precisions) / len(reference_tokens) if reference_tokens else 0

    return average_precision

def calculate_map(generated_captions, reference_captions):
    """
    Calculate the Mean Average Precision (MAP) for all generated captions.
    """
    average_precisions = []
    for filename, generated_caption in generated_captions.items():
        references = reference_captions.get(filename, [])
        ap = calculate_average_precision(generated_caption, references)
        average_precisions.append(ap)

    mean_average_precision = sum(average_precisions) / len(average_precisions) if average_precisions else 0
    return mean_average_precision