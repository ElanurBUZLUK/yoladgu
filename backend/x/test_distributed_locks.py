#!/usr/bin/env python3
"""
Test script for Distributed Locks and Idempotency features
"""
import asyncio
import httpx
import json
import time
from typing import Dict, Any

# Configuration
BASE_URL = "http://localhost:8000"
TEST_USER_EMAIL = "testuser@example.com"
TEST_USER_PASSWORD = "testpass123"

async def test_distributed_locks():
    """Test distributed locks and idempotency features"""
    
    async with httpx.AsyncClient() as client:
        print("🔒 Testing Distributed Locks and Idempotency")
        print("=" * 60)
        
        # Login
        print("\n1. Logging in...")
        login_data = {
            "username": TEST_USER_EMAIL,
            "password": TEST_USER_PASSWORD
        }
        
        try:
            response = await client.post(f"{BASE_URL}/api/v1/auth/login", json=login_data)
            
            if response.status_code == 200:
                auth_data = response.json()
                access_token = auth_data.get("access_token")
                headers = {"Authorization": f"Bearer {access_token}"}
                print("✅ Login successful")
            else:
                print(f"❌ Login failed: {response.status_code} - {response.text}")
                return
                
        except Exception as e:
            print(f"❌ Login error: {e}")
            return
        
        test_failed = False
        
        # Test Vector Management with Distributed Locks
        print("\n2. Testing Vector Management with Distributed Locks...")
        
        # Test batch upsert with lock protection
        try:
            batch_data = {
                "items": [
                    {
                        "obj_ref": "test_question_1",
                        "content": "What is 2 + 2?",
                        "id": "test_1"
                    },
                    {
                        "obj_ref": "test_question_2", 
                        "content": "What is 3 * 4?",
                        "id": "test_2"
                    }
                ],
                "table_name": "questions",
                "namespace": "test_namespace"
            }
            
            response = await client.post(
                f"{BASE_URL}/api/v1/vector/batch-upsert",
                json=batch_data,
                headers=headers
            )
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Batch upsert with lock protection successful")
                print(f"   Processed: {result.get('processed', 0)} items")
                print(f"   Namespace: {result.get('namespace')}")
                print(f"   Slot: {result.get('slot')}")
            else:
                print(f"❌ Batch upsert failed: {response.status_code} - {response.text}")
                test_failed = True
        except Exception as e:
            print(f"❌ Batch upsert error: {e}")
            test_failed = True
        
        # Test concurrent batch upsert (should be blocked by lock)
        print("\n3. Testing Concurrent Operations (Lock Protection)...")
        
        try:
            # Start multiple concurrent requests
            async def concurrent_upsert():
                batch_data = {
                    "items": [
                        {
                            "obj_ref": f"concurrent_test_{time.time()}",
                            "content": "Concurrent test question",
                            "id": f"concurrent_{time.time()}"
                        }
                    ],
                    "table_name": "questions",
                    "namespace": "test_namespace"
                }
                
                response = await client.post(
                    f"{BASE_URL}/api/v1/vector/batch-upsert",
                    json=batch_data,
                    headers=headers
                )
                return response.status_code, response.text
            
            # Run concurrent requests
            tasks = [concurrent_upsert() for _ in range(3)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            print(f"   Concurrent requests completed: {len(results)}")
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    print(f"   Request {i+1}: Exception - {result}")
                else:
                    status_code, text = result
                    print(f"   Request {i+1}: {status_code} - {text[:50]}...")
                    
        except Exception as e:
            print(f"❌ Concurrent test error: {e}")
            test_failed = True
        
        # Test Idempotent Rebuild Index
        print("\n4. Testing Idempotent Rebuild Index...")
        
        try:
            rebuild_data = {
                "namespace": "test_namespace",
                "force": False
            }
            
            response = await client.post(
                f"{BASE_URL}/api/v1/vector/rebuild-index",
                json=rebuild_data,
                headers=headers
            )
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Idempotent rebuild index successful")
                print(f"   Namespace: {result.get('namespace')}")
                print(f"   Questions updated: {result.get('questions_updated', 0)}")
                print(f"   Patterns updated: {result.get('patterns_updated', 0)}")
            else:
                print(f"❌ Rebuild index failed: {response.status_code} - {response.text}")
                test_failed = True
        except Exception as e:
            print(f"❌ Rebuild index error: {e}")
            test_failed = True
        
        # Test concurrent rebuild (should be idempotent)
        print("\n5. Testing Concurrent Idempotent Operations...")
        
        try:
            async def concurrent_rebuild():
                rebuild_data = {
                    "namespace": "test_namespace",
                    "force": False
                }
                
                response = await client.post(
                    f"{BASE_URL}/api/v1/vector/rebuild-index",
                    json=rebuild_data,
                    headers=headers
                )
                return response.status_code, response.text
            
            # Run concurrent rebuild requests
            tasks = [concurrent_rebuild() for _ in range(3)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            print(f"   Concurrent rebuild requests completed: {len(results)}")
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    print(f"   Request {i+1}: Exception - {result}")
                else:
                    status_code, text = result
                    print(f"   Request {i+1}: {status_code} - {text[:50]}...")
                    
        except Exception as e:
            print(f"❌ Concurrent rebuild test error: {e}")
            test_failed = True
        
        # Test Lock Status Checking
        print("\n6. Testing Lock Status Checking...")
        
        try:
            lock_key = "lock:vector:upsert:questions:test_namespace"
            
            response = await client.get(
                f"{BASE_URL}/api/v1/vector/lock-status/{lock_key}",
                headers=headers
            )
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Lock status check successful")
                print(f"   Lock key: {result.get('lock_key')}")
                print(f"   Locked: {result.get('locked')}")
                print(f"   TTL: {result.get('ttl_ms')}ms")
            else:
                print(f"❌ Lock status check failed: {response.status_code} - {response.text}")
                test_failed = True
        except Exception as e:
            print(f"❌ Lock status check error: {e}")
            test_failed = True
        
        # Test Vector Statistics
        print("\n7. Testing Vector Statistics...")
        
        try:
            response = await client.get(
                f"{BASE_URL}/api/v1/vector/statistics",
                headers=headers
            )
            
            if response.status_code == 200:
                stats = response.json()
                print("✅ Vector statistics successful")
                print(f"   Questions: {stats.get('questions', {})}")
                print(f"   Error patterns: {stats.get('error_patterns', {})}")
                print(f"   Index status: {stats.get('index_status', {})}")
            else:
                print(f"❌ Vector statistics failed: {response.status_code} - {response.text}")
                test_failed = True
        except Exception as e:
            print(f"❌ Vector statistics error: {e}")
            test_failed = True
        
        # Test Vector Health Check
        print("\n8. Testing Vector Health Check...")
        
        try:
            response = await client.get(
                f"{BASE_URL}/api/v1/vector/health",
                headers=headers
            )
            
            if response.status_code == 200:
                health = response.json()
                print("✅ Vector health check successful")
                print(f"   Status: {health.get('status')}")
                print(f"   Vector indexes: {health.get('vector_indexes')}")
                print(f"   Redis connection: {health.get('redis_connection')}")
            else:
                print(f"❌ Vector health check failed: {response.status_code} - {response.text}")
                test_failed = True
        except Exception as e:
            print(f"❌ Vector health check error: {e}")
            test_failed = True
        
        # Test Cleanup Slots
        print("\n9. Testing Cleanup Slots...")
        
        try:
            cleanup_data = {
                "namespace": "test_namespace",
                "keep_slots": 2
            }
            
            response = await client.post(
                f"{BASE_URL}/api/v1/vector/cleanup-slots",
                json=cleanup_data,
                headers=headers
            )
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Cleanup slots successful")
                print(f"   Slots cleaned: {result.get('slots_cleaned', 0)}")
                print(f"   Cleaned slots: {result.get('cleaned_slots', [])}")
            else:
                print(f"❌ Cleanup slots failed: {response.status_code} - {response.text}")
                test_failed = True
        except Exception as e:
            print(f"❌ Cleanup slots error: {e}")
            test_failed = True
        
        # Test Manual Rebuild with Context Manager
        print("\n10. Testing Manual Rebuild with Context Manager...")
        
        try:
            response = await client.post(
                f"{BASE_URL}/api/v1/vector/admin/rebuild-index-manual?namespace=test_namespace",
                headers=headers
            )
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Manual rebuild with context manager successful")
                print(f"   Namespace: {result.get('namespace')}")
                print(f"   Total processed: {result.get('total_processed', 0)}")
            elif response.status_code == 409:
                print("✅ Manual rebuild correctly blocked (409 Conflict)")
                print("   Another rebuild is in progress")
            else:
                print(f"❌ Manual rebuild failed: {response.status_code} - {response.text}")
                test_failed = True
        except Exception as e:
            print(f"❌ Manual rebuild error: {e}")
            test_failed = True
        
        # Test Force Release Lock
        print("\n11. Testing Force Release Lock...")
        
        try:
            lock_key = "lock:vector:upsert:questions:test_namespace"
            
            response = await client.delete(
                f"{BASE_URL}/api/v1/vector/lock/{lock_key}",
                headers=headers
            )
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Force release lock successful")
                print(f"   Success: {result.get('success')}")
                print(f"   Message: {result.get('message')}")
            else:
                print(f"❌ Force release lock failed: {response.status_code} - {response.text}")
                test_failed = True
        except Exception as e:
            print(f"❌ Force release lock error: {e}")
            test_failed = True
        
        # Summary
        print("\n" + "=" * 60)
        if test_failed:
            print("❌ Some distributed lock tests failed!")
        else:
            print("✅ All distributed lock tests passed!")
        
        print("\n🎉 Distributed Locks and Idempotency test completed!")

if __name__ == "__main__":
    asyncio.run(test_distributed_locks())
