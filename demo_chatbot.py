#!/usr/bin/env python3
"""
Groundwater Chatbot Demo
========================

This script demonstrates the chatbot's capabilities without the web interface.
Useful for testing and understanding how the chatbot processes queries.
"""

import sys
import os

# Add the frontend directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'frontend'))

from chatbot import GroundwaterChatbot

def demo_queries():
    """Run a series of demo queries to showcase chatbot capabilities"""
    
    print("🤖 Groundwater Assistant Demo")
    print("=" * 50)
    
    # Initialize chatbot
    try:
        chatbot = GroundwaterChatbot()
        print("✅ Chatbot initialized successfully!")
    except Exception as e:
        print(f"❌ Error initializing chatbot: {e}")
        return
    
    # Demo queries
    demo_questions = [
        "Hello! What can you help me with?",
        "What's the rainfall in Maharashtra?",
        "Show me groundwater levels in Delhi",
        "Predict groundwater level for Karnataka with 150mm rainfall",
        "Show rainfall trends in Tamil Nadu",
        "Which states are available?",
        "Help me understand your features"
    ]
    
    print("\n🎯 Running Demo Queries:")
    print("-" * 30)
    
    for i, question in enumerate(demo_questions, 1):
        print(f"\n{i}. User: {question}")
        print("   Bot:", end=" ")
        
        try:
            response = chatbot.generate_response(question)
            # Print response with proper formatting
            lines = response.split('\n')
            for j, line in enumerate(lines):
                if j == 0:
                    print(line)
                else:
                    print("      " + line)
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("-" * 30)
    
    print("\n🎉 Demo completed!")
    print("\n💡 To run the full interactive chatbot:")
    print("   python run_chatbot.py")
    print("   or")
    print("   streamlit run frontend/chatbot.py")

def interactive_demo():
    """Run an interactive demo where user can type queries"""
    
    print("🤖 Interactive Groundwater Assistant Demo")
    print("=" * 50)
    print("Type your questions (or 'quit' to exit)")
    print("Example: 'What's the rainfall in Maharashtra?'")
    print("-" * 50)
    
    # Initialize chatbot
    try:
        chatbot = GroundwaterChatbot()
        print("✅ Chatbot ready!")
    except Exception as e:
        print(f"❌ Error initializing chatbot: {e}")
        return
    
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("👋 Goodbye!")
                break
            
            if not user_input:
                continue
            
            print("🤖 Bot:", end=" ")
            response = chatbot.generate_response(user_input)
            
            # Print response with proper formatting
            lines = response.split('\n')
            for i, line in enumerate(lines):
                if i == 0:
                    print(line)
                else:
                    print("      " + line)
                    
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    """Main function"""
    print("Choose demo mode:")
    print("1. Automated demo (shows example queries)")
    print("2. Interactive demo (type your own queries)")
    print("3. Exit")
    
    while True:
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == '1':
            demo_queries()
            break
        elif choice == '2':
            interactive_demo()
            break
        elif choice == '3':
            print("👋 Goodbye!")
            break
        else:
            print("Please enter 1, 2, or 3")

if __name__ == "__main__":
    main()
