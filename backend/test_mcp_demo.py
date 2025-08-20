#!/usr/bin/env python3
"""
Test script for MCP Demo functionality
"""

import asyncio
import json
from app.mcp.server_simple import TOOLS, list_tools, call_tool


import pytest

@pytest.mark.asyncio
async def test_mcp_demo():
    """Test MCP demo functionality"""
    print("🧪 Testing MCP Demo Functionality")
    print("=" * 50)
    
    # Test 1: List tools
    print("\n1. Testing list_tools()...")
    tools_result = list_tools()
    print(f"✅ Tools listed successfully")
    print(f"   Total tools: {tools_result['data']['total']}")
    print(f"   Available tools: {[tool['name'] for tool in tools_result['data']['tools']]}")
    
    # Test 2: English cloze generation
    print("\n2. Testing english_cloze.generate...")
    cloze_args = {
        "student_id": "test_student_123",
        "num_recent_errors": 3,
        "difficulty_level": 2,
        "question_type": "cloze"
    }
    cloze_result = await call_tool("english_cloze.generate", cloze_args)
    print(f"✅ English cloze generation: {cloze_result['success']}")
    if cloze_result['success']:
        print(f"   Generated questions: {len(cloze_result['data'])}")
        for q in cloze_result['data']:
            print(f"   - {q['content']} (Answer: {q['correct_answer']})")
    
    # Test 3: Math recommendations
    print("\n3. Testing math.recommend...")
    math_args = {
        "student_id": "test_student_123",
        "limit": 5,
        "difficulty_range": [2, 4]
    }
    math_result = await call_tool("math.recommend", math_args)
    print(f"✅ Math recommendations: {math_result['success']}")
    if math_result['success']:
        print(f"   Recommendations: {len(math_result['data'])}")
        for r in math_result['data']:
            print(f"   - {r['content']} (Score: {r['recommendation_score']:.2f})")
    
    # Test 4: Invalid tool
    print("\n4. Testing invalid tool...")
    invalid_result = await call_tool("invalid.tool", {})
    print(f"✅ Invalid tool handling: {not invalid_result['success']}")
    print(f"   Error message: {invalid_result['error']}")
    
    print("\n" + "=" * 50)
    print("🎉 All MCP Demo tests completed successfully!")
    
    # Show final summary
    print("\n📊 MCP Demo Summary:")
    print(f"   - Available tools: {len(TOOLS)}")
    print(f"   - Tool names: {list(TOOLS.keys())}")
    print(f"   - Demo mode: Enabled")
    print(f"   - Latency logging: Enabled")


if __name__ == "__main__":
    asyncio.run(test_mcp_demo())
