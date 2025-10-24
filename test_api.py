"""
API Testing Script for Healthcare Chatbot Backend
"""

import asyncio
import json
import time
from typing import Dict, Any
import httpx
from datetime import datetime

# Configuration
API_BASE_URL = "http://localhost:8000"
TIMEOUT = 30.0

class HealthcareAPITester:
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=TIMEOUT)
        self.test_results = []
    
    async def test_health_check(self) -> Dict[str, Any]:
        """Test health check endpoint"""
        print("\n🔍 Testing Health Check...")
        try:
            response = await self.client.get(f"{self.base_url}/health")
            assert response.status_code == 200
            data = response.json()
            assert "status" in data
            assert data["status"] == "healthy"
            print("✅ Health check passed")
            return {"test": "health_check", "status": "passed", "response": data}
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            return {"test": "health_check", "status": "failed", "error": str(e)}
    
    async def test_analyze_report(self) -> Dict[str, Any]:
        """Test report analysis endpoint"""
        print("\n🔍 Testing Report Analysis...")
        
        test_cases = [
            {
                "name": "Respiratory symptoms",
                "report_text": "Patient is a 45-year-old male presenting with fever of 102°F, persistent dry cough for 5 days, shortness of breath, and fatigue. No known allergies. History of hypertension.",
                "max_conditions": 3
            },
            {
                "name": "Diabetes symptoms",
                "report_text": "55-year-old female with increased thirst, frequent urination, unexplained weight loss of 10 pounds over 2 months, blurred vision, and tingling in feet.",
                "max_conditions": 2
            },
            {
                "name": "Cardiac symptoms",
                "report_text": "68-year-old patient experiencing chest pain radiating to left arm, shortness of breath during exertion, irregular heartbeat, and swelling in ankles.",
                "max_conditions": 3
            }
        ]
        
        results = []
        for test_case in test_cases:
            print(f"\n  📋 Testing: {test_case['name']}")
            try:
                response = await self.client.post(
                    f"{self.base_url}/api/v1/analyze-report",
                    json={
                        "report_text": test_case["report_text"],
                        "max_conditions": test_case["max_conditions"],
                        "include_raw_context": False
                    }
                )
                
                assert response.status_code == 200
                data = response.json()
                
                # Validate response structure
                assert "potential_conditions" in data
                assert "disclaimer" in data
                assert isinstance(data["potential_conditions"], list)
                
                print(f"  ✅ Found {len(data['potential_conditions'])} conditions")
                
                for condition in data["potential_conditions"]:
                    if "disease_name" in condition:
                        confidence = condition.get("confidence_score", "N/A")
                        print(f"     - {condition['disease_name']} (Confidence: {confidence})")
                
                results.append({
                    "test_case": test_case["name"],
                    "status": "passed",
                    "conditions_found": len(data["potential_conditions"])
                })
                
            except Exception as e:
                print(f"  ❌ Failed: {e}")
                results.append({
                    "test_case": test_case["name"],
                    "status": "failed",
                    "error": str(e)
                })
        
        return {"test": "analyze_report", "results": results}
    
    async def test_chat_functionality(self) -> Dict[str, Any]:
        """Test chat endpoint"""
        print("\n🔍 Testing Chat Functionality...")
        
        test_questions = [
            "What are the early symptoms of diabetes?",
            "How is pneumonia diagnosed?",
            "What treatments are available for hypertension?",
            "Can you explain the difference between Type 1 and Type 2 diabetes?",
            "What preventive measures can be taken against heart disease?"
        ]
        
        conversation_id = f"test_conv_{int(time.time())}"
        results = []
        
        for question in test_questions:
            print(f"\n  💬 Question: {question[:50]}...")
            try:
                response = await self.client.post(
                    f"{self.base_url}/api/v1/chat",
                    json={
                        "message": question,
                        "conversation_id": conversation_id
                    }
                )
                
                assert response.status_code == 200
                data = response.json()
                
                # Validate response
                assert "response" in data
                assert "conversation_id" in data
                assert len(data["response"]) > 0
                
                print(f"  ✅ Response received ({len(data['response'])} chars)")
                
                if "sources" in data and data["sources"]:
                    print(f"  📚 Sources: {', '.join([s['disease'] for s in data['sources'][:3]])}")
                
                results.append({
                    "question": question[:50],
                    "status": "passed",
                    "response_length": len(data["response"])
                })
                
            except Exception as e:
                print(f"  ❌ Failed: {e}")
                results.append({
                    "question": question[:50],
                    "status": "failed",
                    "error": str(e)
                })
        
        # Test conversation retrieval
        print("\n  📜 Testing conversation history retrieval...")
        try:
            response = await self.client.get(
                f"{self.base_url}/api/v1/conversation/{conversation_id}"
            )
            assert response.status_code == 200
            data = response.json()
            assert "messages" in data
            print(f"  ✅ Retrieved {len(data['messages'])} messages")
        except Exception as e:
            print(f"  ❌ Conversation retrieval failed: {e}")
        
        return {"test": "chat", "conversation_id": conversation_id, "results": results}
    
    async def test_error_handling(self) -> Dict[str, Any]:
        """Test error handling"""
        print("\n🔍 Testing Error Handling...")
        
        test_cases = [
            {
                "name": "Empty report",
                "endpoint": "/api/v1/analyze-report",
                "method": "POST",
                "data": {"report_text": ""},
                "expected_status": 422
            },
            {
                "name": "Invalid conversation ID",
                "endpoint": "/api/v1/conversation/invalid_id_12345",
                "method": "GET",
                "data": None,
                "expected_status": 404
            },
            {
                "name": "Too long report",
                "endpoint": "/api/v1/analyze-report",
                "method": "POST",
                "data": {"report_text": "x" * 15000},
                "expected_status": 422
            }
        ]
        
        results = []
        for test_case in test_cases:
            print(f"\n  🧪 Testing: {test_case['name']}")
            try:
                if test_case["method"] == "POST":
                    response = await self.client.post(
                        f"{self.base_url}{test_case['endpoint']}",
                        json=test_case["data"]
                    )
                else:
                    response = await self.client.get(
                        f"{self.base_url}{test_case['endpoint']}"
                    )
                
                if response.status_code == test_case["expected_status"]:
                    print(f"  ✅ Correctly returned status {response.status_code}")
                    results.append({
                        "test_case": test_case["name"],
                        "status": "passed"
                    })
                else:
                    print(f"  ⚠️ Unexpected status: {response.status_code}")
                    results.append({
                        "test_case": test_case["name"],
                        "status": "warning",
                        "actual_status": response.status_code
                    })
                    
            except Exception as e:
                print(f"  ❌ Error: {e}")
                results.append({
                    "test_case": test_case["name"],
                    "status": "failed",
                    "error": str(e)
                })
        
        return {"test": "error_handling", "results": results}
    
    async def test_performance(self) -> Dict[str, Any]:
        """Test API performance"""
        print("\n🔍 Testing Performance...")
        
        test_report = "Patient presents with headache, fever, and body aches."
        
        # Single request timing
        print("\n  ⏱️ Single request timing...")
        start = time.time()
        response = await self.client.post(
            f"{self.base_url}/api/v1/analyze-report",
            json={"report_text": test_report}
        )
        single_time = time.time() - start
        print(f"  ✅ Single request: {single_time:.2f}s")
        
        # Concurrent requests
        print("\n  ⏱️ Testing 5 concurrent requests...")
        start = time.time()
        tasks = [
            self.client.post(
                f"{self.base_url}/api/v1/analyze-report",
                json={"report_text": test_report}
            )
            for _ in range(5)
        ]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        concurrent_time = time.time() - start
        
        successful = sum(1 for r in responses if not isinstance(r, Exception) and r.status_code == 200)
        print(f"  ✅ Concurrent requests: {concurrent_time:.2f}s ({successful}/5 successful)")
        
        return {
            "test": "performance",
            "single_request_time": single_time,
            "concurrent_time": concurrent_time,
            "concurrent_success_rate": successful / 5
        }
    
    async def run_all_tests(self):
        """Run all tests"""
        print("=" * 60)
        print("🏥 Healthcare Chatbot API Testing Suite")
        print("=" * 60)
        print(f"Target: {self.base_url}")
        print(f"Started: {datetime.now().isoformat()}")
        
        # Run tests
        tests = [
            self.test_health_check(),
            self.test_analyze_report(),
            self.test_chat_functionality(),
            self.test_error_handling(),
            self.test_performance()
        ]
        
        results = await asyncio.gather(*tests, return_exceptions=True)
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        
        passed = 0
        failed = 0
        
        for result in results:
            if isinstance(result, Exception):
                print(f"❌ Test crashed: {result}")
                failed += 1
            elif isinstance(result, dict):
                test_name = result.get("test", "Unknown")
                if "results" in result:
                    # Multiple sub-tests
                    sub_passed = sum(1 for r in result["results"] if r.get("status") == "passed")
                    sub_failed = sum(1 for r in result["results"] if r.get("status") == "failed")
                    print(f"📝 {test_name}: {sub_passed} passed, {sub_failed} failed")
                    passed += sub_passed
                    failed += sub_failed
                else:
                    # Single test
                    status = result.get("status", "unknown")
                    if status == "passed":
                        print(f"✅ {test_name}: PASSED")
                        passed += 1
                    else:
                        print(f"❌ {test_name}: FAILED")
                        failed += 1
        
        print(f"\nTotal: {passed} passed, {failed} failed")
        print("=" * 60)
        
        # Save results
        report_file = f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "base_url": self.base_url,
                "results": [r for r in results if isinstance(r, dict)],
                "summary": {
                    "passed": passed,
                    "failed": failed
                }
            }, f, indent=2, default=str)
        
        print(f"\n📄 Test report saved to: {report_file}")
        
        await self.client.aclose()

async def main():
    """Main test runner"""
    tester = HealthcareAPITester()
    await tester.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())