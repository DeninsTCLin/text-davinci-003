import streamlit as st
import openai
import re
import nltk
nltk.download('punkt')
from nltk import tokenize
from pysummarization.nlpbase.auto_abstractor import AutoAbstractor
from pysummarization.tokenizabledoc.simple_tokenizer import SimpleTokenizer
from pysummarization.abstractabledoc.top_n_rank_abstractor import TopNRankAbstractor

# nltk.download('omw-1.4')

api_key = st.text_input(label='Enter your API Key',)


# def clean_text(string):
#     new_string = re.sub('[^a-zA-Z0-9 \n\.]','',string)
#     return new_string
def clean_text(doc):
    # p = re.compile(r"^Speaker$", re.IGNORECASE)
    # cleaned_doc = p.sub(' ', doc)
    cleaned_doc = re.sub(r'\d+', '', doc)
    cleaned_doc = re.sub('[^A-Za-z0-9]+', ' ', doc)
    return cleaned_doc

def summarize(doc):
    # Object of automatic summarization.
    auto_abstractor = AutoAbstractor()
    # Set tokenizer.
    auto_abstractor.tokenizable_doc = SimpleTokenizer()
    # Set delimiter for making a list of sentence.
    auto_abstractor.delimiter_list = [".", "\n"]
    # Object of abstracting and filtering document.
    abstractable_doc = TopNRankAbstractor()
    result_dict = auto_abstractor.summarize(doc, abstractable_doc)
    summary = result_dict["summarize_result"]
    summary = ' '.join(summary)
    return summary

def generate(text,userPrompt="Extract key insights:"):
    openai.api_key = api_key
    response = openai.Completion.create(
    engine="text-davinci-003",
    prompt=userPrompt + "\n\n" + text,
    temperature=0.7,
    max_tokens=250,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    for r in response['choices']:
        print(r['text'])
    return response


def display_app_header(main_txt, sub_txt, is_sidebar=False):
    """
    Code Credit: https://github.com/soft-nougat/dqw-ivves
    function to display major headers at user interface
    :param main_txt: the major text to be displayed
    :param sub_txt: the minor text to be displayed
    :param is_sidebar: check if its side panel or major panel
    :return:
    """
    html_temp = f"""
    <h2 style = "text_align:center; font-weight: bold;"> {main_txt} </h2>
    <p style = "text_align:center;"> {sub_txt} </p>
    </div>
    """
    if is_sidebar:
        st.sidebar.markdown(html_temp, unsafe_allow_html=True)
    else:
        st.markdown(html_temp, unsafe_allow_html=True)

def divider():
    """
    Sub-routine to create a divider for webpage contents
    """
    st.markdown("""---""")


@st.cache
def clean_txt(doc):
    return clean_text(doc)

@st.cache
def summary(doc):
    return summarize(doc)

@st.cache
def generate_insights(text,userPrompt="Extract key insights:"):
    return generate(text,userPrompt="Extract key insights:")

def main():
    st.write("""
    # GPT-3 Text Processing Demo
    """)
    input_help_text = """
    Enter Text
    """
    final_message = """
    The data was successfully analyzed
    """
    text = st.text_area(label='INPUT TEXT',placeholder="Enter Sample Text")
    # text = clean_txt(text)
    text = summary(text)
    # st.write(text)
    


    with st.sidebar:
        st.markdown("**Processing**")
        first_summary = st.button(
            label="Extract Insights",
            help=""
        )
    #     # second_summary = st.button(
    #     #     label="TL;DR summarization",
    #     #     help=""
    #     # )
    #     # third_summary = st.button(
    #     #     label="Notes to summary",
    #     #     help=""
    #     # )
    #     # final_summary = st.button(
    #     #     label="Final summary",
    #     #     help=""
    #     # )

    if first_summary:
        st.markdown("#### Key Insights")
        with st.spinner('Wait for it...'):
            output = generate_insights(text).get("choices")[0]['text']
            # st.write(output)
            text_sentences = tokenize.sent_tokenize(output)
            for sentence in text_sentences:
                st.write('•',sentence)
            # for body in text:
            #     output = generate_insights(body).get("choices")[0]['text']
            #     text_sentences = tokenize.sent_tokenize(output)
            #     for sentence in text_sentences:
            #         st.write('•',sentence)

    # if second_summary:
    #     st.markdown("#### TL;DR Summary")
    #     with st.spinner('Wait for it...'):
    #         output2 = tldr_summary(text).get("choices")[0]['text']
    #         text_sentences = tokenize.sent_tokenize(output2)
    #         for sentence in text_sentences:
    #             st.write('•',sentence)

    # if third_summary:
    #     st.markdown("#### Notes to summary")
    #     with st.spinner('Wait for it...'):
    #         output3 = note_summary(text).get("choices")[0]['text']
    #         text_sentences = tokenize.sent_tokenize(output3)
    #         for sentence in text_sentences:
    #             st.write('•',sentence)

    # if final_summary:
    #     st.markdown("#### Final summary")
    #     with st.spinner('Wait for it...'):
    #         output1 = second(text).get("choices")[0]['text']
    #         output2 = tldr_summary(text).get("choices")[0]['text']
    #         output3 = note_summary(text).get("choices")[0]['text']
    #         output = tldr_summary(output1 + output2).get("choices")[0]['text']
    #         text_sentences = tokenize.sent_tokenize(output)
    #         for sentence in text_sentences:
    #             st.write('•',sentence)

if __name__ == '__main__':
    main()
