#!/usr/bin/env python3
"""
Dashboard API test script
"""
import asyncio
import httpx
import json

BASE_URL = "http://localhost:8000"

async def test_dashboard_apis():
    """Test dashboard and subject selection APIs"""
    
    async with httpx.AsyncClient() as client:
        print("🧪 Testing Dashboard APIs...")
        
        # First, login to get token
        print("\n1. Logging in...")
        login_data = {
            "username": "testuser",
            "password": "testpass123"
        }
        
        try:
            response = await client.post(f"{BASE_URL}/api/v1/users/login", json=login_data)
            if response.status_code == 200:
                token_data = response.json()
                access_token = token_data["access_token"]
                headers = {"Authorization": f"Bearer {access_token}"}
                print("✅ Login successful")
            else:
                print(f"❌ Login failed: {response.status_code}")
                return
        except Exception as e:
            print(f"❌ Login error: {e}")
            return
        
        # Test subject selection data
        print("\n2. Testing subject selection data...")
        try:
            response = await client.get(f"{BASE_URL}/api/v1/dashboard/subject-selection", headers=headers)
            if response.status_code == 200:
                print("✅ Subject selection data retrieved")
                selection_data = response.json()
                print(f"   Available subjects: {selection_data['available_subjects']}")
                print(f"   User levels: {selection_data['user_levels']}")
                print(f"   Recommendations: {len(selection_data['recommendations'])} items")
            else:
                print(f"❌ Subject selection failed: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"❌ Subject selection error: {e}")
        
        # Test subject selection
        print("\n3. Testing subject selection...")
        try:
            select_data = {
                "subject": "math",
                "selected_at": "2024-01-01T12:00:00Z"
            }
            response = await client.post(f"{BASE_URL}/api/v1/dashboard/select-subject", json=select_data, headers=headers)
            if response.status_code == 200:
                print("✅ Subject selection successful")
                result = response.json()
                print(f"   Selected subject: {result['selected_subject']}")
                print(f"   Message: {result['message']}")
            else:
                print(f"❌ Subject selection failed: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"❌ Subject selection error: {e}")
        
        # Test dashboard data
        print("\n4. Testing dashboard data...")
        try:
            response = await client.get(f"{BASE_URL}/api/v1/dashboard/", headers=headers)
            if response.status_code == 200:
                print("✅ Dashboard data retrieved")
                dashboard_data = response.json()
                print(f"   User: {dashboard_data['user_info']['username']}")
                print(f"   Learning style: {dashboard_data['user_info']['learning_style']}")
                print(f"   Overall accuracy: {dashboard_data['overall_stats']['accuracy_percentage']:.1f}%")
                print(f"   Recommendations: {len(dashboard_data['recommendations'])} items")
            else:
                print(f"❌ Dashboard data failed: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"❌ Dashboard data error: {e}")
        
        # Test math-specific dashboard
        print("\n5. Testing math dashboard...")
        try:
            response = await client.get(f"{BASE_URL}/api/v1/dashboard/math", headers=headers)
            if response.status_code == 200:
                print("✅ Math dashboard retrieved")
                math_data = response.json()
                if math_data['math_progress']:
                    print(f"   Math level: {math_data['math_progress']['current_level']}")
                    print(f"   Progress: {math_data['math_progress']['progress_percentage']:.1f}%")
            else:
                print(f"❌ Math dashboard failed: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"❌ Math dashboard error: {e}")
        
        # Test learning style adaptation
        print("\n6. Testing learning style adaptation...")
        try:
            response = await client.get(f"{BASE_URL}/api/v1/dashboard/learning-style", headers=headers)
            if response.status_code == 200:
                print("✅ Learning style adaptation retrieved")
                adaptation = response.json()
                print(f"   Learning style: {adaptation['learning_style']}")
                print(f"   Recommended features: {len(adaptation['recommended_features'])} items")
                print(f"   UI adaptations: {len(adaptation['adaptations'])} settings")
            else:
                print(f"❌ Learning style adaptation failed: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"❌ Learning style adaptation error: {e}")
        
        # Test performance summary
        print("\n7. Testing performance summary...")
        try:
            response = await client.get(f"{BASE_URL}/api/v1/dashboard/performance/math?period=week", headers=headers)
            if response.status_code == 200:
                print("✅ Performance summary retrieved")
                performance = response.json()
                print(f"   Subject: {performance['subject']}")
                print(f"   Period: {performance['period']}")
                print(f"   Questions attempted: {performance['questions_attempted']}")
                print(f"   Accuracy rate: {performance['accuracy_rate']:.1f}%")
            else:
                print(f"❌ Performance summary failed: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"❌ Performance summary error: {e}")
        
        # Test weekly progress
        print("\n8. Testing weekly progress...")
        try:
            response = await client.get(f"{BASE_URL}/api/v1/dashboard/weekly-progress", headers=headers)
            if response.status_code == 200:
                print("✅ Weekly progress retrieved")
                weekly = response.json()
                print(f"   Week: {weekly['week_start']} to {weekly['week_end']}")
                print(f"   Questions answered: {weekly['questions_answered']}")
                print(f"   Total time spent: {weekly['total_time_spent']} seconds")
                print(f"   Daily stats: {len(weekly['daily_stats'])} days")
            else:
                print(f"❌ Weekly progress failed: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"❌ Weekly progress error: {e}")
        
        # Test stats overview
        print("\n9. Testing stats overview...")
        try:
            response = await client.get(f"{BASE_URL}/api/v1/dashboard/stats/overview", headers=headers)
            if response.status_code == 200:
                print("✅ Stats overview retrieved")
                overview = response.json()
                print(f"   Math level: {overview['math_level']}")
                print(f"   English level: {overview['english_level']}")
                print(f"   Recent activities: {overview['recent_activity_count']}")
                print(f"   Achievements: {overview['achievements_count']}")
            else:
                print(f"❌ Stats overview failed: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"❌ Stats overview error: {e}")
        
        # Test recommendations
        print("\n10. Testing recommendations...")
        try:
            response = await client.get(f"{BASE_URL}/api/v1/dashboard/recommendations?limit=3", headers=headers)
            if response.status_code == 200:
                print("✅ Recommendations retrieved")
                recommendations = response.json()
                print(f"   Recommendations: {len(recommendations['recommendations'])} items")
                for i, rec in enumerate(recommendations['recommendations'][:3], 1):
                    print(f"   {i}. {rec}")
            else:
                print(f"❌ Recommendations failed: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"❌ Recommendations error: {e}")
        
        print("\n🎉 Dashboard API test completed!")

if __name__ == "__main__":
    asyncio.run(test_dashboard_apis())