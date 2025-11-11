#!/usr/bin/env python3
"""
Quick test script to verify chatbot functionality
"""

import os
import sys

def test_file_paths():
    """Test if all required files exist"""
    print("🔍 Testing file paths...")
    
    required_files = [
        "data/rainfall.csv",
        "data/groundwater.csv", 
        "data/regions.geojson",
        "models/groundwater_predictor.pkl"
    ]
    
    all_good = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - NOT FOUND")
            all_good = False
    
    return all_good

def test_chatbot_import():
    """Test if chatbot can be imported"""
    print("\n🔍 Testing chatbot import...")
    
    try:
        # Add frontend to path and import
        sys.path.insert(0, os.path.join(os.getcwd(), 'frontend'))
        from chatbot import GroundwaterChatbot
        print("✅ Chatbot import successful")
        
        # Test initialization
        print("🔍 Testing chatbot initialization...")
        chatbot = GroundwaterChatbot()
        print("✅ Chatbot initialization successful")
        
        # Test basic functionality
        print("🔍 Testing basic functionality...")
        states = chatbot.get_available_states()
        print(f"✅ Found {len(states)} states")
        
        # Test a simple query
        response = chatbot.generate_response("Hello")
        print("✅ Response generation successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 Groundwater Chatbot Test Suite")
    print("=" * 40)
    
    # Test file paths
    files_ok = test_file_paths()
    
    if files_ok:
        # Test chatbot functionality
        chatbot_ok = test_chatbot_import()
        
        if chatbot_ok:
            print("\n🎉 All tests passed! Chatbot is ready to use.")
            print("\nTo run the chatbot:")
            print("  python run_chatbot.py")
            print("  or")
            print("  cd frontend && streamlit run chatbot.py")
        else:
            print("\n❌ Chatbot tests failed. Check the errors above.")
    else:
        print("\n❌ File path tests failed. Make sure all required files exist.")

if __name__ == "__main__":
    main()
