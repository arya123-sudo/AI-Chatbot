# ============================================
# GEMINI AI CHATBOT
# ============================================

from google import genai

# ============================================
# GEMINI CLIENT
# ============================================

client = genai.Client(
    api_key="YOUR_API_KEY"
)
# ============================================
# CHATBOT LOOP
# ============================================

print("===================================")
print(" GEMINI AI LEARNING ASSISTANT")
print("===================================")
print("Type 'bye' to exit")
print()

while True:

    user_input = input("You : ")

    if user_input.lower() == "bye":

        print("Bot : Goodbye!")

        break

    try:

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=user_input
        )

        print("\nBot :", response.text)

    except Exception as e:

        print("Error :", e)