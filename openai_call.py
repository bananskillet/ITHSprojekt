import openai
import settings

openai.api_key = settings.OPENAI_API_KEY

def generate_summarization(article_text, title, key_phrases, keywords, sentiment, lengt_of_summarization):

    response = openai.ChatCompletion.create(model="gpt-4",
                                    messages=[{"role": "system", "content": f"""You are a expert in {keywords}.
                                               You should make a output text that should be in the format as a article or section. The title for the input text is ({title}) and is a guidence for the whole output.
                                               the most important things in the text, take in the consideration the phrases in the bracket ({key_phrases}). 
                                               The sentiment of the overal output should be {sentiment}. Focus on figures if found in the text. The output should be maximum {lengt_of_summarization} words."""},
                                    {"role": "user", "content": article_text}])

    summary = response["choices"][0]["message"]["content"]

    return summary
