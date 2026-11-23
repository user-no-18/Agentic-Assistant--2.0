import asyncio

from dotenv import load_dotenv

from agent import build_agent


async def main() -> None:
    load_dotenv()
    agent = build_agent()
    thread_id = "cli-session"

    print("Chatbot ready! Type '/bye' to exit.\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "/bye":
            break
        if not user_input:
            continue

        try:
            result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": user_input}]},
                {"configurable": {"thread_id": thread_id}},
            )
            last_message = result["messages"][-1]
            print(f"AI: {last_message.content}\n")
        except Exception as exc:
            print(f"Error: {exc}\n")


if __name__ == "__main__":
    asyncio.run(main())
