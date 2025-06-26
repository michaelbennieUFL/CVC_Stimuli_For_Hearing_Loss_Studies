import math


def combine_samples(sample1, sample2):
    # based off of https://handbook-5-1.cochrane.org/chapter_7/table_7_7_a_formulae_for_combining_groups.htm
    # Unpack inputs
    n1, m1, sd1 = sample1
    n2, m2, sd2 = sample2

    # Combined sample size
    n_combined = n1 + n2

    # Combined mean
    mean_combined = (n1 * m1 + n2 * m2) / n_combined

    # Combined variance
    term1 = (n1 - 1) * sd1**2
    term2 = (n2 - 1) * sd2**2
    correction = (n1 * n2) / n_combined * (m1 - m2)**2
    variance_combined = (term1 + term2 + correction) / (n_combined - 1)

    # Combined standard deviation
    sd_combined = math.sqrt(variance_combined)

    return n_combined, mean_combined, sd_combined


if __name__ == "__main__":
    sample1 = (12, 327, 41)
    sample2 = (12, 290, 10)

    combined = combine_samples(sample1, sample2)
    print(f"Combined sample: n={combined[0]}, mean={combined[1]:.2f}, sd={combined[2]:.2f}")
