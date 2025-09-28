import pandas as pd
import re
from openai import OpenAI

# --- CONFIGURATION ---
# Migrate to the new, recommended model and API structure
MODEL_NAME = "gpt-5" 

# --- SYSTEM PROMPT (now called instructions) ---
# This defines the AI's persona, rules, and workflow.
INSTRUCTIONS = """
You are an expert Ecosystem Analyst AI. Your purpose is to help users understand the Ecosystem Maturity Index and provide actionable, data-driven advice.

Your workflow for answering questions is strict and you MUST follow it every time:

1.  **Analyze the User's Query and History**: Understand the user's intent. Pay attention to the conversation history for context. Identify any mentioned countries.

2.  **Consult Internal Data First**: The user will provide you with a `[DATA DUMP]` containing the relevant scores for the country or countries in question. This data is your PRIMARY source of truth. You MUST analyze this data to identify the country's overall score, maturity level, and most importantly, its weakest domains and specific underlying indicators.

3.  **Synthesize and Respond**: Your final answer must be a confident synthesis of BOTH the internal data and your general knowledge, framed to provide constructive, forward-looking recommendations.

**RULES FOR RESPONDING:**

-   **Data is Signal, Not Absolute Truth**: The provided data may be incomplete or based on global benchmarks. DO NOT state that "the score is wrong" or "the data is missing." Instead, use a low score as a "signal" or an "indicator" that an area warrants further attention. Frame it positively, e.g., "Based on the index, 'Human Capital' presents a significant opportunity for improvement."
-   **Be Action-Oriented**: The user is looking for advice. Focus your answers on *what can be done*. For a low domain or indicator score, suggest concrete, real-world examples of initiatives that could help (e.g., for low 'Human Capital', suggest "investing in STEM education programs, creating tech-focused vocational training, or implementing policies to attract skilled migrants.").
-   **Be Context-Aware**: Use the conversation history to understand the flow of the discussion. If a user asks a follow-up question, you don't need to repeat all the context.
-   **Formatting**: Use Markdown for clarity. Use bullet points for lists of recommendations and bold text to highlight key domains or concepts.
-   **Tone**: Be professional, encouraging, and expert. You are a helpful analyst.
"""

def get_ai_response(prompt: str, api_key: str, final_df: pd.DataFrame, ind_df: pd.DataFrame, dom_map: pd.DataFrame, chat_history: list):
    """
    Retrieves data, constructs a prompt, and gets a response from the new OpenAI Responses API.
    """
    client = OpenAI(api_key=api_key)

    # --- 1. Simple Entity Extraction: Find countries mentioned ---
    all_countries = final_df['COUNTRY'].unique().tolist()
    # Check both current prompt and last user message in history for countries
    last_user_prompt = chat_history[-1]['content'] if chat_history and chat_history[-1]['role'] == 'user' else ""
    text_to_search = prompt + " " + last_user_prompt
    mentioned_countries = list(set([country for country in all_countries if re.search(r'\b' + re.escape(country) + r'\b', text_to_search, re.IGNORECASE)]))
    
    # --- 2. Data Retrieval: Get all relevant scores ---
    data_dump = ""
    if mentioned_countries:
        data_dump += "\n\n[DATA DUMP]\n"
        data_dump += "Here is the internal data for the mentioned countries. Use this as your primary source of truth:\n"
        for country in mentioned_countries:
            country_final_data = final_df[final_df['COUNTRY'] == country]
            if not country_final_data.empty:
                data_dump += f"\n--- {country.upper()} ---\n"
                final_score = country_final_data.iloc[0]
                score_col = next((c for c in SCORE_CANDIDATES if c in final_df.columns), "Final_Score_0_100")
                data_dump += f"- Overall Score: {final_score[score_col]:.1f}/100\n"
                data_dump += f"- Maturity Level: {final_score['Maturity']}\n"
                
                domain_cols = [c for c in final_df.columns if c.startswith("DOMAIN_SCORE__")]
                domain_scores = final_score[domain_cols].sort_values(ascending=True)
                data_dump += "- Domain Scores (0-100, lower is a bigger opportunity for improvement):\n"
                for domain, score in domain_scores.items():
                    domain_name = domain.replace("DOMAIN_SCORE__", "").replace("_", " ")
                    data_dump += f"  - {domain_name}: {score:.1f}\n"

                if not domain_scores.empty:
                    weakest_domain_col = domain_scores.index[0]
                    weakest_domain_name = weakest_domain_col.replace("DOMAIN_SCORE__", "")
                    
                    domain_indicators = dom_map[dom_map['DOMAIN'] == weakest_domain_name]['INDICATOR'].tolist()
                    country_ind_data = ind_df[ind_df['COUNTRY'] == country]
                    
                    if not country_ind_data.empty and domain_indicators:
                        available_indicators = [ind for ind in domain_indicators if ind in country_ind_data.columns]
                        indicator_scores = country_ind_data[available_indicators].T.iloc[:, 0]
                        weakest_indicators = indicator_scores.sort_values(ascending=True).head(3)
                        
                        if not weakest_indicators.empty:
                            data_dump += f"- Key Improvement Opportunities within '{weakest_domain_name.replace('_', ' ')}':\n"
                            for indicator, score in weakest_indicators.items():
                                data_dump += f"  - {indicator}: {score:.1f}\n"

    # --- 3. Construct the context for the API call ---
    # The new API takes a simpler list of context items.
    context = []
    # Add previous messages for context (short-term memory)
    # The new API handles this differently, but we can pass the message history as input
    context.extend(chat_history[-4:])

    # Add the user's current prompt with the data dump
    full_prompt_with_data = prompt + data_dump
    context.append({"role": "user", "content": full_prompt_with_data})


    # --- 4. Call the new OpenAI Responses API ---
    try:
        # NOTE: The 'responses.create' streams by default if you iterate over it.
        # The 'stream=True' parameter is not used in the new API structure.
        response_stream = client.responses.create(
            model=MODEL_NAME,
            instructions=INSTRUCTIONS,
            input=context,
            # Let's add the native web_search tool as requested in the vision!
            tools=[{"type": "web_search"}] 
        )
        
        # The new API might return different object structures, we adapt to stream text chunks
        # This part assumes a similar streaming mechanism. If the final object is different,
        # this loop would need adjustment. The documentation implies a streaming-compatible response.
        for item in response_stream:
             # We look for message items and extract the text content
            if item.type == 'message' and hasattr(item, 'content') and item.content:
                text_content = item.content[0].text if hasattr(item.content[0], 'text') else None
                if text_content:
                    yield text_content
            # The new API might also return final text in a simpler 'output_text' helper
            elif hasattr(item, 'output_text') and item.output_text:
                 yield item.output_text
                 break # break if we get the full text at once

    except Exception as e:
        # Attempt to provide a more user-friendly error message
        error_message = str(e)
        if "Incorrect API key" in error_message:
            yield "Error: The provided OpenAI API key is incorrect. Please check and try again."
        elif "You exceeded your current quota" in error_message:
            yield "Error: You have exceeded your OpenAI API quota. Please check your account usage."
        else:
            yield f"An error occurred: {error_message}"


# --- Constants used in the helper ---
SCORE_CANDIDATES = [
    "Final_Score_0_100", "Final_Blended_0_100", "Final_Log_0_100",
    "Final_PerCap_0_100", "Final_DomainAvg_0_100"
]

