import os
import asyncio
from dotenv import load_dotenv
from lab1_agent.graph import app

# Load env variables (NVIDIA_API_KEY)
load_dotenv()

async def main():
    print("--- NVIDIA Solutions Architect AI Assistant (Lab 1) ---")
    print("Type 'quit' to exit.")
    
    if not os.getenv("NVIDIA_API_KEY"):
        print("ERROR: NVIDIA_API_KEY not found in environment. Please set it in .env")
        return

    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ["quit", "exit"]:
            break
            
        print("\nAssistant is thinking...", end="", flush=True)
        
        # Stream events from the graph
        # This shows the "Thought Process" (Router -> Tool -> Generator)
        inputs = {"question": user_input}
        async for output in app.astream(inputs, stream_mode="updates"):
            for key, value in output.items():
                print(f"\n--- Node '{key}' finished ---")
                if key in ["nvidia_tutor", "general_tutor"]:
                    print(f"\nFinal Answer: \n{value['generation']}")
                elif key == "start_hardware_calc":
                     print(f"Calculated: {value['generation']}")

if __name__ == "__main__":
    asyncio.run(main())
