import asyncio
import os
from playwright.async_api import async_playwright

async def main():
    print("Starting Playwright recording...")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        # Record video into the current directory
        context = await browser.new_context(
            record_video_dir=".", 
            record_video_size={"width": 1280, "height": 720}
        )
        page = await context.new_page()
        
        print("Navigating to localhost:8502...")
        try:
            await page.goto("http://localhost:8502")
        except Exception as e:
            print(f"Failed to load page. Is the frontend running? Error: {e}")
            await context.close()
            await browser.close()
            return

        # Wait for the chat to actually render
        await page.wait_for_timeout(2000)
        
        print("Typing question...")
        # The input has placeholder "Ask a question (e.g..."
        await page.wait_for_selector('input[type="text"]')
        await page.fill('input[type="text"]', "Explain what Triton is in one sentence.")
        
        await page.wait_for_timeout(1000)
        print("Sending message...")
        await page.click('button.send-btn')
        
        print("Waiting for agent to respond (15 seconds)...")
        # Wait 15 seconds to capture the thinking animation and the final response
        await page.wait_for_timeout(15000)
        
        video_path = await page.video.path()
        print(f"Video recorded temporarily at: {video_path}")
        
        await context.close()
        await browser.close()
        
        # Rename the video to demo.webm
        if os.path.exists(video_path):
            os.rename(video_path, "demo.webm")
            print("Successfully saved recording to demo.webm!")

if __name__ == "__main__":
    asyncio.run(main())
