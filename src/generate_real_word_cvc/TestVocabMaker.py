import CMUreader
from tabulate import tabulate
from CMUreader import parse_cmudict, generateCombinedWordsets, generateWordCategoryLists, load_unique_words, \
    filter_cmudict_words


#TODO:
# talk aboout AO sound



def generatePlosiveInitalTests(word_set) -> dict:
    wordCategoryList = generateWordCategoryLists()
    all_plosive_initals = {}

    for plosive in wordCategoryList["Consonants_Stops"]:
        plosive_initals = CMUreader.filterByLetters(word_set, [plosive])
        plosive_initals = CMUreader.filterByPhonemeCount(plosive_initals, 3, int.__eq__)
        plosive_initals = CMUreader.filterByLetters(plosive_initals, wordCategoryList["Consonants_All"], index=-1)
        plosive_initals = CMUreader.filterByLetters(plosive_initals, wordCategoryList["Vowels_Monophthong_Uncentralized_NonMid"], index=-2)
        plosive_initals = CMUreader.filterBySyllableCount(plosive_initals, 1, int.__eq__)
        all_plosive_initals[plosive] = plosive_initals

    return all_plosive_initals

def generatePlosiveFinalsTests(word_set) -> dict:
    wordCategoryList = generateWordCategoryLists()
    all_plosive_finals = {}

    for plosive in wordCategoryList["Consonants_Stops"]:
        plosive_finals = CMUreader.filterByLetters(word_set, [plosive], index=-1)
        plosive_finals = CMUreader.filterByPhonemeCount(plosive_finals, 3, int.__eq__)
        plosive_finals = CMUreader.filterByLetters(plosive_finals, wordCategoryList["Consonants_All"], index=0)
        plosive_finals = CMUreader.filterByLetters(plosive_finals, wordCategoryList["Vowels_Monophthong_Uncentralized_NonMid"], index=-2)
        plosive_finals = CMUreader.filterBySyllableCount(plosive_finals, 1, int.__eq__)
        all_plosive_finals[plosive] = plosive_finals

    return all_plosive_finals


def generateUnvoicedAfterSTests(word_set) -> dict:
    wordCategoryList = generateWordCategoryLists()
    all_unvoiced_plosives_after_s = {}

    for plosive in wordCategoryList["Consonants_Stops_Unvoiced"]:
        unvoiced_plosives_after_s = CMUreader.filterByLetters(word_set, wordCategoryList["Consonants_S"], index=0)
        unvoiced_plosives_after_s = CMUreader.filterBySyllableCount(unvoiced_plosives_after_s, 1, int.__eq__)
        unvoiced_plosives_after_s = CMUreader.filterByPhonemeCount(unvoiced_plosives_after_s, 4, int.__eq__)
        unvoiced_plosives_after_s = CMUreader.filterByLetters(unvoiced_plosives_after_s, [plosive], index=1)
        unvoiced_plosives_after_s = CMUreader.filterByLetters(unvoiced_plosives_after_s, wordCategoryList["Consonants_All"], index=-1)
        unvoiced_plosives_after_s = CMUreader.filterByLetters(unvoiced_plosives_after_s, wordCategoryList["Vowels_Monophthong_Uncentralized_NonMid"], index=-2)
        all_unvoiced_plosives_after_s[plosive] = unvoiced_plosives_after_s

    return all_unvoiced_plosives_after_s


def generateDevoicedTests(word_set) -> dict:
    wordCategoryList = generateWordCategoryLists()
    all_devoiced_plosives = {}


    unvoiced_non_plosives=list(
        set(wordCategoryList["Consonants_Unvoiced"]) - set(wordCategoryList["Consonants_Stops"]))


    for plosive in wordCategoryList["Consonants_Stops_Voiced"]:

        devoiced_plosives = CMUreader.filterBySyllableCount(word_set, 2, int.__eq__)
        devoiced_plosives = CMUreader.filterByLetterPair(devoiced_plosives,wordCategoryList["Consonants_All"], wordCategoryList["Vowels_Monophthong_Uncentralized_NonMid"])
        devoiced_plosives = CMUreader.filterByLetterPair(devoiced_plosives, wordCategoryList["Vowels_Monophthong_Uncentralized_NonMid"],[plosive])
        devoiced_plosives=CMUreader.filterByLetterPair(devoiced_plosives,[plosive],unvoiced_non_plosives)
        all_devoiced_plosives[plosive] = devoiced_plosives

    return all_devoiced_plosives


def generateVCVPlosiveTests(word_set) -> dict:
    wordCategoryList = generateWordCategoryLists()
    all_plosive_middles = {}

    for plosive in wordCategoryList["Consonants_Stops"]:
        plosive_middle = CMUreader.filterByPhonemeCount(word_set, 3, int.__eq__)
        plosive_middle = CMUreader.filterByLetters(plosive_middle, [plosive], index=1)
        plosive_middle = CMUreader.filterByLetters(plosive_middle, wordCategoryList["Vowels_All"], index=0)
        plosive_middle = CMUreader.filterByLetters(plosive_middle, wordCategoryList["Vowels_All"], index=-1)
        #plosive_middle = CMUreader.filterBySyllableCount(plosive_middle, 1, int.__eq__)
        all_plosive_middles[plosive] = plosive_middle

    return all_plosive_middles


def generateVCVFillers(word_set) -> dict:
    wordCategoryList = generateWordCategoryLists()
    all_plosive_middles = []

    for plosive in wordCategoryList["Consonants_Stops_NONE"]:
        plosive_middle = CMUreader.filterByPhonemeCount(word_set, 3, int.__eq__)
        plosive_middle = CMUreader.filterByLetters(plosive_middle, [plosive], index=1)
        plosive_middle = CMUreader.filterByLetters(plosive_middle, wordCategoryList["Vowels_All"], index=0)
        plosive_middle = CMUreader.filterByLetters(plosive_middle, wordCategoryList["Vowels_All"], index=-1)
        #plosive_middle = CMUreader.filterBySyllableCount(plosive_middle, 1, int.__eq__)
        if plosive_middle:
            all_plosive_middles.extend(plosive_middle)

    return {"fillers":all_plosive_middles}

def generateSCVCFillers(word_set) -> dict:
    wordCategoryList = generateWordCategoryLists()
    all_unvoiced_plosives_after_s = []

    for plosive in wordCategoryList["Consonants_Stops_NONE"]:
        unvoiced_plosives_after_s = CMUreader.filterByLetters(word_set, wordCategoryList["Consonants_S"], index=0)
        unvoiced_plosives_after_s = CMUreader.filterBySyllableCount(unvoiced_plosives_after_s, 1, int.__eq__)
        unvoiced_plosives_after_s = CMUreader.filterByPhonemeCount(unvoiced_plosives_after_s, 4, int.__eq__)
        unvoiced_plosives_after_s = CMUreader.filterByLetters(unvoiced_plosives_after_s, [plosive], index=1)
        unvoiced_plosives_after_s = CMUreader.filterByLetters(unvoiced_plosives_after_s, wordCategoryList["Consonants_All"], index=-1)
        unvoiced_plosives_after_s = CMUreader.filterByLetters(unvoiced_plosives_after_s, wordCategoryList["Vowels_Monophthong_Uncentralized_NonMid"], index=-2)
        if unvoiced_plosives_after_s:
            all_unvoiced_plosives_after_s.extend(unvoiced_plosives_after_s)

    return {"fillers":all_unvoiced_plosives_after_s}

def generateAllCVCFillers(word_set) -> dict:
    wordCategoryList = generateWordCategoryLists()
    all_plosive_initals = []

    for plosive in wordCategoryList["Consonants_Stops_NONE"]:
        plosive_initals = CMUreader.filterByLetters(word_set, [plosive])
        plosive_initals = CMUreader.filterByPhonemeCount(plosive_initals, 3, int.__eq__)
        plosive_initals = CMUreader.filterByLetters(plosive_initals, wordCategoryList["Consonants_Stops_NONE"], index=-1)
        plosive_initals = CMUreader.filterByLetters(plosive_initals, wordCategoryList["Vowels_Monophthong_Uncentralized_NonMid"], index=-2)
        plosive_initals = CMUreader.filterBySyllableCount(plosive_initals, 1, int.__eq__)
        if plosive_initals:
            all_plosive_initals.extend(plosive_initals)

    return {"fillers":all_plosive_initals}


def generateAllCVC(word_set) -> dict:
    wordCategoryList = generateWordCategoryLists()
    all_plosive_initals = {}

    for plosive in wordCategoryList["Consonants_All"]:
        plosive_initals = CMUreader.filterByLetters(word_set, [plosive])
        plosive_initals = CMUreader.filterByPhonemeCount(plosive_initals, 3, int.__eq__)
        plosive_initals = CMUreader.filterByLetters(plosive_initals, wordCategoryList["Consonants_All"], index=-1)
        plosive_initals = CMUreader.filterByLetters(plosive_initals, wordCategoryList["Vowels_All"], index=-2)
        plosive_initals = CMUreader.filterBySyllableCount(plosive_initals, 1, int.__eq__)
        if plosive_initals:
            all_plosive_initals[plosive] = plosive_initals

    return all_plosive_initals

def generateBisyllabicFillers(word_set) -> dict:
    wordCategoryList = generateWordCategoryLists()
    all_devoiced_plosives = []


    unvoiced_non_plosives=list(
        set(wordCategoryList["Consonants_Unvoiced"]) - set(wordCategoryList["Consonants_Stops"]))


    for plosive in wordCategoryList["Consonants_Stops_NONE"]:

        devoiced_plosives = CMUreader.filterBySyllableCount(word_set, 2, int.__eq__)
        devoiced_plosives = CMUreader.filterByLetterPair(devoiced_plosives,wordCategoryList["Consonants_Stops_NONE"], wordCategoryList["Vowels_Monophthong_Uncentralized_NonMid"])
        devoiced_plosives = CMUreader.filterByLetterPair(devoiced_plosives, wordCategoryList["Vowels_Monophthong_Uncentralized_NonMid"],[plosive])
        devoiced_plosives=CMUreader.filterByLetterPair(devoiced_plosives,[plosive],unvoiced_non_plosives)
        if devoiced_plosives:
            all_devoiced_plosives.extend(devoiced_plosives)

    return {"fillers":all_devoiced_plosives}

def generate_markdown_table(test_word_list):
    # Define test types and map them to the dataset
    test_types = [
        ("CVC", test_word_list["CVC"]),
    ]

    # Phonemes to include as columns
    phonemes = ["P", "T", "K", "B", "D", "G"]

    # Prepare table data
    table_data = []
    for test_name, data in test_types:
        row = [test_name]  # First column is the test type
        for ph in phonemes:
            row.append(len(data.get(ph, [])) if ph in data else "-")  # Get word count or "-"
        table_data.append(row)

    # Generate Markdown table
    markdown_table = tabulate(table_data, headers=["Test Set Type \\ Phoneme"] + phonemes, tablefmt="github")
    print(markdown_table)  # Print to console
    return markdown_table  # Return as a string if needed elsewhere




def generateTestWordList(word_set):
    result_sets = {

        "CVC": generateAllCVC(word_set),
    }



    # Print the results for Plosive Starts
    print("\nCVC  Tests:")
    for plosive, words in result_sets["CVC"].items():
        print(f"{plosive}: {len(words)} words")



    generate_markdown_table(result_sets)

    return result_sets



if __name__ == "__main__":
    dictLocation = "../../input_data/"
    word_categories = generateWordCategoryLists()
    print("\nGenerated Vowel Category (Monophthong Uncentralized Non-Mid):")
    print(word_categories["Vowels_Monophthong_Uncentralized_NonMid"])

    original_word_set=generateCombinedWordsets(dictLocation)

    unique_l2_words=load_unique_words(dictLocation+"dictionaries/Oxford_3000_5000_AmericanEnglish.txt")

    filtered_word_set=filter_cmudict_words(original_word_set,unique_l2_words)
    filter_cmudict_words
    result = generateTestWordList(filtered_word_set)
    print(result)
