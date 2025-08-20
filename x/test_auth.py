#!/usr/bin/env python3
"""
Simple authentication test script
"""
import asyncio
import httpx
import json

import pytest

BASE_URL = "http://localhost:8000"

@pytest.mark.asyncio
async def test_authentication():
    """Test authentication flow"""
    
    async with httpx.AsyncClient() as client:
        print("🧪 Testing Authentication System...")
        
        # Test user registration
        print("\n1. Testing user registration...")
        register_data = {
            "username": "testuser",
            "email": "test@example.com",
            "password": "testpass123",
            "role": "student",
            "learning_style": "mixed"
        }
        
        try:
            response = await client.post(f"{BASE_URL}/api/v1/users/register", json=register_data)
            if response.status_code == 201:
                print("✅ User registration successful")
                user_data = response.json()
                print(f"   User ID: {user_data['id']}")
                print(f"   Username: {user_data['username']}")
            else:
                print(f"❌ Registration failed: {response.status_code} - {response.text}")
                return
        except Exception as e:
            print(f"❌ Registration error: {e}")
            return
        
        # Test user login
        print("\n2. Testing user login...")
        login_data = {
            "username": "testuser",
            "password": "testpass123"
        }
        
        try:
            response = await client.post(f"{BASE_URL}/api/v1/users/login", json=login_data)
            if response.status_code == 200:
                print("✅ User login successful")
                token_data = response.json()
                access_token = token_data["access_token"]
                print(f"   Token type: {token_data['token_type']}")
                print(f"   Expires in: {token_data['expires_in']} seconds")
            else:
                print(f"❌ Login failed: {response.status_code} - {response.text}")
                return
        except Exception as e:
            print(f"❌ Login error: {e}")
            return
        
        # Test protected endpoint
        print("\n3. Testing protected endpoint...")
        headers = {"Authorization": f"Bearer {access_token}"}
        
        try:
            response = await client.get(f"{BASE_URL}/api/v1/users/me", headers=headers)
            if response.status_code == 200:
                print("✅ Protected endpoint access successful")
                user_info = response.json()
                print(f"   Current user: {user_info['username']}")
                print(f"   Role: {user_info['role']}")
                print(f"   Learning style: {user_info['learning_style']}")
            else:
                print(f"❌ Protected endpoint failed: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"❌ Protected endpoint error: {e}")
        
        # Test user update
        print("\n4. Testing user update...")
        update_data = {
            "learning_style": "visual",
            "current_math_level": 3,
            "current_english_level": 2
        }
        
        try:
            response = await client.put(f"{BASE_URL}/api/v1/users/me", json=update_data, headers=headers)
            if response.status_code == 200:
                print("✅ User update successful")
                updated_user = response.json()
                print(f"   New learning style: {updated_user['learning_style']}")
                print(f"   Math level: {updated_user['current_math_level']}")
                print(f"   English level: {updated_user['current_english_level']}")
            else:
                print(f"❌ User update failed: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"❌ User update error: {e}")
        
        # Test password reset
        print("\n5. Testing password reset...")
        reset_data = {"email": "test@example.com"}
        
        try:
            response = await client.post(f"{BASE_URL}/api/v1/users/password-reset", json=reset_data)
            if response.status_code == 200:
                print("✅ Password reset request successful")
                reset_response = response.json()
                reset_token = reset_response.get("token")  # Only for testing
                
                if reset_token:
                    # Test password reset confirmation
                    confirm_data = {
                        "token": reset_token,
                        "new_password": "newpass123"
                    }
                    
                    response = await client.post(f"{BASE_URL}/api/v1/users/password-reset/confirm", json=confirm_data)
                    if response.status_code == 200:
                        print("✅ Password reset confirmation successful")
                    else:
                        print(f"❌ Password reset confirmation failed: {response.status_code}")
            else:
                print(f"❌ Password reset failed: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"❌ Password reset error: {e}")
        
        # Test user stats
        print("\n6. Testing user statistics...")
        try:
            response = await client.get(f"{BASE_URL}/api/v1/users/me/stats", headers=headers)
            if response.status_code == 200:
                print("✅ User stats retrieval successful")
                stats = response.json()
                print(f"   Total attempts: {stats['total_attempts']}")
                print(f"   Accuracy rate: {stats['accuracy_rate']}%")
            else:
                print(f"❌ User stats failed: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"❌ User stats error: {e}")
        
        print("\n🎉 Enhanced authentication test completed!")

if __name__ == "__main__":
    asyncio.run(test_authentication())