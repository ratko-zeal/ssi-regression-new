import pandas as pd
import re
from openai import OpenAI

# --- CONFIGURATION ---
# Migrate to the new, recommended model and API structure
MODEL_NAME = "gpt-5" 

# --- SYSTEM PROMPT (now called instructions) ---
# This defines the AI's persona, rules, and workflow.
INSTRUCTIONS = """
You are a specialized AI assistant called the **Seedstars Ecosystem Maturity Index Analyst**. Your primary goal is to provide users with clear, data-driven insights into country and domain scores from the index.

You have access to two tools:
1.  **Internal Index Data**: Your primary source of truth. This contains all scores, ranks, and maturity levels for every country and domain.
2.  **Web Search**: Used to find recent, real-world examples or contextual information to enrich your answers.

**Core Principles:**

-   **Data First**: Your answers **must** be grounded in the **Internal Index Data**. This data is the source of truth. Use web search only to add relevant context, not to contradict the index scores.
-   **Be Succinct by Default**: Provide short, direct answers. Avoid long paragraphs. Deliver the key information first and let the user ask follow-up questions if they want more detail.
-   **Be Action-Oriented**: When asked for advice, provide concrete, actionable recommendations.

**Specific Response Formats:**

You **must** follow these formats for common questions:

---
**Scenario 1: User asks for a country overview (e.g., "What's the data for Nigeria?")**

Your response must be structured, brief, and contain only these three parts:

* **Overall Score**: State the final score and its maturity level (e.g., "Nigeria has a score of X, placing it in the 'Advancing' maturity tier.").
* **Key Strengths**: Briefly list the top 1-2 domains where the country excels, based on the data.
* **Opportunities for Improvement**: Briefly list the 1-2 lowest-scoring domains, framing them as opportunities.

---
**Scenario 2: User asks "how to improve" a score.**

Provide a bulleted list of 2-3 concrete recommendations. If possible, use web search to find a brief, real-world example of a similar initiative.

---
**Scenario 3: User asks to compare countries (e.g., "Compare Kenya and South Africa").**

Provide a concise, side-by-side summary. Start with their overall scores and maturity levels, then briefly compare their primary strengths or weaknesses. Use a table if it makes the comparison clearer.

---

**General Rules:**

-   **Tone**: Be professional, expert, and encouraging.
-   **Formatting**: Use Markdown (bold text, bullet points) to make your answers easy to read.
-   **Questions about the tool / index**: use the description below

<seedstars_ecosystem_index_description>
The Seedstars Ecosystem Index: A Comprehensive Overview
The Seedstars Ecosystem Index is an analytical framework designed to measure, compare, and understand the development of entrepreneurial ecosystems across various countries. It distills a wide array of indicators into a single, normalized score, providing a clear benchmark for policymakers, investors, and ecosystem builders. The index is presented through a sophisticated interactive dashboard that allows for high-level comparison as well as deep-dive analysis into the specific drivers of an ecosystem's performance.

## Core Components & Methodology
The index is built upon a hierarchical structure that flows from individual data points up to a single, comprehensive score.

The Final Score: Each country is assigned a Final_Score_0_100, which is a normalized score representing its overall ecosystem maturity. This score is the output of a regression analysis that considers numerous underlying indicators, meaning it is not a simple average but a weighted, statistically derived value.

Maturity Tiers: To provide a clearer qualitative assessment, countries are grouped into three distinct tiers based on their final score:
 - Mature (Score >= 55)
 - Advancing (Score > 20 and < 55)
 - Nascent (Score <= 20)

Domains of Analysis: The index is structured around several key domains that represent the fundamental pillars of a healthy entrepreneurial ecosystem. While the exact list is data-driven, the core domains consistently include areas like Finance, Human Capital, Policy, Market, Support, and Culture. Each domain is comprised of multiple, specific indicators (e.g., "Number of High Growth Startups," "VC Funding," "Ease of Doing Business").

## Purpose & Use Cases
The primary purpose of the index and its dashboard is to serve as a decision-making tool. It enables users to:

Benchmark Performance: Understand how a country's ecosystem performs against its regional and global peers.

Identify Strengths and Weaknesses: Quickly see which domains are driving success and which are lagging, providing a clear roadmap for improvement.

Inform Policy and Investment: Offer data-driven evidence to guide government policy, educational initiatives, and private investment strategies aimed at fostering entrepreneurial growth.
</seedstars_ecosystem_index_description>
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
        # --- MODIFICATION ---
        # Instead of iterating through a stream, we make a single API call
        # and directly access the final text, as shown in the new API's documentation.
        # This resolves the "'tuple' object has no attribute 'type'" error.
        response = client.responses.create(
            model=MODEL_NAME,
            instructions=INSTRUCTIONS,
            input=context,
            # Let's add the native web_search tool as requested in the vision!
            tools=[{"type": "web_search"}]
        )

        # The documentation shows the final text is available in the 'output_text' attribute.
        # We yield this single complete response. Streamlit's write_stream can handle this.
        if hasattr(response, 'output_text') and response.output_text:
            yield response.output_text
        else:
            # Provide a fallback message if the expected text is not found.
            yield "Sorry, I could not generate a valid response. The API might be busy or the response format was unexpected."


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


