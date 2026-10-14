import os
import chromadb
import streamlit as st 
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# page configuration
st.set_page_config(
    page_title="MoodCare",
    page_icon="💙",
    layout="centered"
)

# custom styling
st.html(
    """
    <style>

    /* =========================================
       1. MAIN DARK THEME
       ========================================= */

    .stApp {
        background: #0f1117 !important;
        color: #e8edf7 !important;
    }


    /* =========================================
       2. MAIN CONTENT AREA
       ========================================= */

    .block-container {
        max-width: 760px !important;
        padding-top: 2.5rem !important;
        padding-bottom: 7rem !important;
    }


    /* =========================================
       3. MOODCARE HEADER
       ========================================= */

    .moodcare-title {
        text-align: center;
        font-size: 40px;
        font-weight: 700;
        color: #f1f5ff !important;
        margin: 8px 0 6px 0;
        letter-spacing: -1px;
    }

    .moodcare-subtitle {
        text-align: center;
        font-size: 15px;
        color: #8d99ad !important;
        margin: 0 0 34px 0;
    }

    
    /* =========================================
       4. CHAT AREA
       ========================================= */

    .moodcare-chat {
        display: flex;
        flex-direction: column;
        gap: 14px;
    }


    /* =========================================
       5. USER MESSAGE
       ========================================= */

    .moodcare-user-row {
        display: flex;
        justify-content: flex-end;
        width: 100%;
    }

    .moodcare-user-bubble {
        display: flex;
        align-items: flex-start;
        gap: 10px;
        max-width: 72%;
        padding: 11px 14px;
        background: #1d2940;
        border: 1px solid #2b3c5c;
        border-radius: 18px 18px 5px 18px;
        color: #e8edf7;
        font-size: 15px;
        line-height: 1.5;
        box-shadow: 0 3px 12px rgba(0, 0, 0, 0.18);
    }


    /* =========================================
       6. ASSISTANT MESSAGE
       ========================================= */

    .moodcare-assistant-row {
        display: flex;
        justify-content: flex-start;
        width: 100%;
    }

    .moodcare-assistant-bubble {
        display: flex;
        align-items: flex-start;
        gap: 10px;
        max-width: 76%;
        padding: 11px 14px;
        background: #171b24;
        border: 1px solid #262d3a;
        border-radius: 18px 18px 18px 5px;
        color: #e8edf7;
        font-size: 15px;
        line-height: 1.5;
        box-shadow: 0 3px 12px rgba(0, 0, 0, 0.20);
    }


    /* =========================================
       7. AVATARS
       ========================================= */

    .moodcare-avatar {
        width: 30px;
        height: 30px;
        min-width: 30px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 10px;
        font-size: 16px;
        line-height: 1; ////
    }

    .moodcare-user-avatar {
        background: #273654;
    }

    .moodcare-assistant-avatar {
        background: #172b4d;
    }


    /* =========================================
       8. MESSAGE TEXT
       ========================================= */

    .moodcare-message-text {
        padding-top: 3px;
        word-break: break-word; ////
    }


    /* =========================================
       9. CHAT INPUT BACKGROUND
       ========================================= */

    [data-testid="stBottomBlockContainer"],
    .stBottomBlockContainer {
        background: #0f1117 !important;
        border: none !important;
        box-shadow: none !important;
    }


    /* =========================================
       10. CHAT INPUT
       ========================================= */

    [data-testid="stChatInput"],
    .stChatInput {
        background: transparent !important;
        padding-bottom: 18px !important;
    }


    /* =========================================
       11. INPUT BOX
       ========================================= */

    [data-testid="stChatInput"] > div {
        background: #171b24 !important;
        border: 1px solid #303847 !important;
        border-radius: 20px !important;
        box-shadow: 0 5px 20px rgba(0, 0, 0, 0.25) !important;
    }


    /* =========================================
       12. INPUT WHEN ACTIVE
       ========================================= */

    [data-testid="stChatInput"] > div:focus-within {
        border-color: #4f7fd8 !important;
        box-shadow: 0 5px 22px rgba(60, 110, 200, 0.16) !important;
    }


    /* =========================================
       13. INPUT TEXT
       ========================================= */

    [data-testid="stChatInput"] textarea {
        background: transparent !important;
        color: #e8edf7 !important;
        -webkit-text-fill-color: #e8edf7 !important;
        caret-color: #6d9cff !important;
        font-size: 15px !important;
    }


    /* =========================================
       14. INPUT PLACEHOLDER
       ========================================= */

    [data-testid="stChatInput"] textarea::placeholder {
        color: #78849a !important;
        -webkit-text-fill-color: #78849a !important;
        opacity: 1 !important;
    }


    /* =========================================
       15. SEND BUTTON
       ========================================= */
    
    /* SEND BUTTON */

    [data-testid="stChatInput"] button {
        background-color: #ffffff !important;
        border: none !important;
        border-radius: 12px !important;
        box-shadow: none !important;
    }

    /* Remove the square outline around the arrow icon */

    [data-testid="stChatInput"] button svg rect {
        display: none !important;
    }

    /* Keep the arrow blue */

    [data-testid="stChatInput"] button svg {
        color: #2563eb !important;
    }

    
    /* =========================================
       16. HIDE STREAMLIT DECORATIONS
       ========================================= */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }


    /* =========================================
       17. MOBILE
       ========================================= */

    @media (max-width: 700px) {

        .block-container {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }

        .moodcare-title {
            font-size: 34px;
        }

        .moodcare-user-bubble,
        .moodcare-assistant-bubble {
            max-width: 86%;
        }
    }

    </style>
    """
)

# moodcare header
st.markdown(
    '<div class="moodcare-title">💙 MoodCare</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="moodcare-subtitle">'
    'A friendly space to talk, feel heard and brighten your day.'
    '</div>',
    unsafe_allow_html=True
)


# Chroma vector database
client = chromadb.PersistentClient(path="vector_db")

collection = client.get_or_create_collection(
    name="moodcare_knowledge"
)

# Initialize the collection if it is empty
if collection.count() == 0:
    import pandas as pd

    # Load the knowledge dataset
    df = pd.read_csv("data/moodcare_knowledge.csv")

    # Load the embedding model
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Combine category, user state and advice
    df["combined_text"] = (
        "Category: " + df["category"].astype(str)
        + " | User State: " + df["user_state"].astype(str)
        + " | Advice: " + df["advice"].astype(str)
    )

    # Generate embeddings
    embeddings = embedding_model.encode(
        df["combined_text"].tolist(),
        show_progress_bar=False
    )

    # Insert knowledge into ChromaDB
    collection.add(
        ids=df["id"].astype(str).tolist(),
        documents=df["combined_text"].tolist(),
        embeddings=embeddings.tolist(),
        metadatas=[
            {
                "category": row["category"],
                "activity_type": row["activity_type"]
            }
            for _, row in df.iterrows()
        ]
    )


# embedding model
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedding_model = load_embedding_model()

# groq client
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# retrieve context
def retrieve_context(query, top_k=3):
    query_embedding = embedding_model.encode(query).tolist()

    results = collection.query(
        query_embeddings = [query_embedding],
        n_results=top_k
    )

    documents = results["documents"][0]

    context = "\n\n".join(documents)

    return context


# generate moodcare response
def generate_response_stream(user_query):

    context = retrieve_context(user_query)

    system_prompt = f"""
You are MoodCare, a friendly, warm and conversational AI companion.

Your purpose is to support users who are feeling lonely, sad,
stressed, anxious, bored, unmotivated, angry or emotionally low.

You should behave like a supportive friend, not like a therapist
or a formal advice-giving system.

Your priority is natural conversation.

Do not treat every conversation as a problem that needs to be solved.
Sometimes the user simply wants to chat, share something ordinary,
or talk about their interests.

IMPORTANT RESPONSE RULES:

1. First understand the user's state and acknowledge it naturally when appropriate.

2. If the user is looking for emotional support or advice, then use the retrieved
   context as the primary source of guidance.

3. Do NOT dump a list of suggestions.
   Prefer one or two useful suggestions that fit the situation.

4. MoodCare can also help users take their mind off things.
   When appropriate, offer choices such as:
   - talking about their day
   - sharing a joke
   - sharing an interesting or surprising fact
   - giving a small fun challenge or activity
   - simply listening

5. If the user explicitly asks for a joke, give them a joke.
   Do not give emotional advice unless it is necessary.

6. If the user explicitly asks for an interesting or crazy fact,
   provide only a factual and well-established fact.

7. Never invent, guess or fabricate a fact.

8. If you are not confident that a fact is accurate, then don't
   present it as a fact.

9. Do not make up statistics, historical events, scientific claims,
   names, dates or other factual details.

10. Keep interesting facts short, clear and genuinely interesting.

11. If the user wants to talk, respond conversationally and invite
    them to continue talking when appropriate.

12. Keep responses warm, natural and concise.
    Usually respond in 1-3 short sentences.

13. Do not give long explanations unless the user explicitly asks
    for more detail.

14. In normal conversation, focus on continuing the conversation
    rather than giving advice or multiple suggestions.

15. Ask at most one natural follow-up question when appropriate.
    A follow-up question is optional, not required.

16. Do not repeatedly offer jokes, facts, activities or other options
    after every message.

17. Match the user's energy and message length.

18. If the user is sharing something positive, engage with it naturally.

19. If the user is emotionally overwhelmed, prioritize empathy,
    listening and conversation.

20. Do not mention retrieved context, vector databases, embeddings,
    Chroma or technical implementation.

21. Do not sound clinical, robotic or overly formal.

22. Do not overwhelm the user with too many options.

23. Maintain conversational continuity. Always relate the response
    to the user's latest message and previous conversation.

24. Do not invent or assume personal details, experiences, feelings,
    events or situations that the user has not mentioned.

25. Do not interpret short replies such as "yes", "many", "okay",
    "nothing", "hmm" or "really?" as emotions without using the
    previous conversation to understand them.

26. When the user clarifies or corrects something, acknowledge the
    clarification and continue from that meaning.

27. If the user is discussing normal topics such as music, movies,
    studies, hobbies or daily activities, stay on that topic naturally.

28. If the user appears to be ending the conversation with "thanks",
    "bye", "goodbye" or "let it be", respond naturally and do not
    force a new topic.

29. If the user's message is ambiguous, use the previous conversation
    before asking for clarification.

30. Do not over-analyze ordinary conversation or assign hidden emotions
    to neutral statements.

31. Prefer natural conversational responses over repetitive supportive
    phrases like "I hear you" when they are not necessary.

32. Do not turn the conversation into an interview. Avoid asking
    follow-up questions after every user message. Questions are
    optional. Often, a natural acknowledgment or conversational
    reaction is enough.

33. Avoid repeating the same observation, question or wording across
    consecutive responses. If the user has already explained something,
    do not ask them to explain the same thing again.

34. A response does not always need a question. When a natural
    acknowledgment is enough, do not force a follow-up question.

35. Do not claim personal experiences, preferences, feelings or
    real-world experiences of your own. Stay warm and conversational
    without pretending to have a personal life.

36. Before asking a question, check the conversation history to see
    whether the user has already answered or discussed that point.
    If they have, acknowledge it and move the conversation forward
    without asking the same question again.

37. Vary conversational wording naturally. Do not repeatedly begin
    responses with phrases such as "Sounds like", "Got it", "Glad to hear",
    or "That sounds".

38. When the user says "what?", "what do you mean?",
    "means what?", or similar phrases, explain the previous
    message in simpler words instead of repeating the same sentence.

39. When the user provides information that answers a previous
    question, acknowledge and use that information. Do not ask
    the same question again in a different form.

40. Do not assume that a phrase refers to a specific fact about
    the user when the meaning is ambiguous. Use the conversation
    context and ask for clarification only when it is genuinely
    necessary.

41. Never describe yourself as having a personal life, day, playlist,
    memories, experiences, tastes, preferences or feelings.
    Do not say things like "my favorite", "my playlist",
    "a song that's been brightening my day", "I watched", or
    "I love this movie". Stay warm without pretending to be a person.

42. If you do not have reliable current information about a topic,
    do not compensate by inventing or guessing specific names,
    titles, dates, releases, statistics or other details.

43. When the user asks about current, recent or future information
    such as movies, music releases, news, products or events,
    only provide specific factual details when reliable current
    information is available. Otherwise clearly say that you are
    not able to verify the latest information.

44. Never provide specific movie titles, cast members, actors,
    actresses, release years, directors, plot details or other
    entertainment facts unless you are confident they are accurate.

45. If the user asks for recent, current-year or upcoming movies
    and reliable current information is not available, do not
    invent titles or cast information. Clearly state that the
    information cannot be verified.

46. Never guess an actor's or actress's involvement in a movie.
    If unsure, do not associate that person with the movie.

47. When correcting a previous factual mistake, do not guess a
    replacement fact. Only provide the correction if it is known
    with high confidence.

48. If the user challenges the accuracy of a factual statement,
    acknowledge the uncertainty and do not generate another
    unverified factual claim to replace it.

49. When recommending multiple items, check the conversation history
    for items that have already been recommended. Do not repeat an
    item that MoodCare has already suggested unless the user
    specifically asks about it again.

50. When the user asks for "more", "another", "more options" or
    similar wording, provide genuinely new options rather than
    repeating items already mentioned in the conversation.

51. When the user explicitly asks for movie recommendations, be
    helpful, but never invent movie titles, actors, release years,
    ratings, reviews, plots or other movie information.

52. When the user asks specifically for current-year, recent or
    upcoming movies and reliable current information is not available,
    clearly say that the latest titles cannot be verified. Do not
    generate substitute titles from uncertain memory.

53. Never create a fictional movie title merely to satisfy a request
    for recommendations.

54. If the user asks for recommendations without requiring current
    releases, only recommend movies whose title, release year and
    basic genre you know with high confidence.

55. If you are uncertain whether a movie exists or whether its details
    are correct, do not recommend it as a factual movie.

56. Do not invent descriptions, reviews, ratings, cast members,
    directors or plot details for movies.

57. If reliable movie information is unavailable, be honest about
    the limitation rather than filling the response with plausible
    but unverified movie recommendations.

58. When recommending multiple items, check the conversation history
    for items that have already been recommended. Do not repeat an
    item that MoodCare has already suggested unless the user
    specifically asks about it again.

59. When the user asks for "more", "another", "more options" or
    similar wording, provide genuinely new options rather than
    repeating items already mentioned in the conversation.

Retrieved_context for emotional-support situations: 
{context}
"""

    messages = [
        {
            "role": "system",
            "content": system_prompt
        }
    ]

    # add previous conversation history
    messages.extend(st.session_state.messages)

    response = groq_client.chat.completions.create(
        model = "openai/gpt-oss-20b",
        messages = messages,
        temperature= 0.7,
        stream = True
    )

    for chunk in response:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

# chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


# display conversation history
st.html('<div class="moodcare-chat">')

for message in st.session_state.messages:

    # check whether the message belongs to the user
    if message["role"] == "user":

        user_text = (
            message["content"]
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

        st.html(
            f"""
            <div class="moodcare-user-row">
                <div class="moodcare-user-bubble">

                    <div class="moodcare-avatar moodcare-user-avatar">
                        😊
                    </div>

                    <div class="moodcare-message-text">
                        {user_text}
                    </div>

                </div>
            </div>
            """
        )

    # otherwise the message belongs to MoodCare
    else:

        assistant_text = (
            message["content"]
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

        st.html(
            f"""
            <div class="moodcare-assistant-row">
                <div class="moodcare-assistant-bubble">

                    <div class="moodcare-avatar moodcare-assistant-avatar">
                        💙
                    </div>

                    <div class="moodcare-message-text">
                        {assistant_text}
                    </div>

                </div>
            </div>
            """
        )

st.html("</div>")


# chat input
user_query = st.chat_input(
    "Hey! How are you feeling today?"
)



# process user message
if user_query:

    # save user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_query
        }
    )

    # escape HTML characters in the user message
    user_text = (
        user_query
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )

    # create a placeholder for the user bubble
    user_placeholder = st.empty()

    # immediately render the user message using existing CSS
    user_placeholder.markdown(
        f"""
        <div class="moodcare-user-row">
            <div class="moodcare-user-bubble">
                <div class="moodcare-avatar moodcare-user-avatar">
                    😊
                </div>
                <div class="moodcare-message-text">
                    {user_text}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # create a placeholder for MoodCare's response
    assistant_placeholder = st.empty()

    # initially show the thinking indicator in the assistant bubble
    assistant_placeholder.markdown(
        """
        <div class="moodcare-assistant-row">
            <div class="moodcare-assistant-bubble">
                <div class="moodcare-avatar moodcare-assistant-avatar">
                    💙
                </div>
                <div class="moodcare-message-text">
                    Thinking... 💭
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # generate and display MoodCare's response progressively
    response_text = ""

    for chunk in generate_response_stream(user_query):

        response_text += chunk

        # escape HTML characters in the generated response
        assistant_text = (
            response_text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "<br>")
        )

        # update the same assistant bubble as tokens arrive
        assistant_placeholder.markdown(
            f"""
            <div class="moodcare-assistant-row">
                <div class="moodcare-assistant-bubble">
                    <div class="moodcare-avatar moodcare-assistant-avatar">
                        💙
                    </div>
                    <div class="moodcare-message-text">
                        {assistant_text}
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # save the complete assistant response
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response_text
        }
    )

    # refresh the interface to show the saved conversation
    st.rerun()
