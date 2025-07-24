#!/usr/bin/env python3
"""
HackRx 6.0 Complete Testing Suite
Tests all components of the system
"""

import requests
import json
import time
import asyncio
from typing import Dict, List

# Test Configuration
BASE_URL = "http://localhost:8000"  # Change to your deployed URL
AUTH_TOKEN = "95f763f2e367cc7e5f72304cb9e9b84229f97f2a5b2b08f14b5034e8328596ec"

# Sample test data from HackRx problem statement
TEST_DOCUMENT_URL = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"

TEST_QUESTIONS = [
    "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
    "What is the waiting period for pre-existing diseases (PED) to be covered?", 
    "Does this policy cover maternity expenses, and what are the conditions?",
    "What is the waiting period for cataract surgery?",
    "Are the medical expenses for an organ donor covered under this policy?",
    "What is the No Claim Discount (NCD) offered in this policy?",
    "Is there a benefit for preventive health check-ups?",
    "How does the policy define a 'Hospital'?",
    "What is the extent of coverage for AYUSH treatments?",
    "Are there any sub-limits on room rent and ICU charges for Plan A?"
]

EXPECTED_ANSWERS = [
    "A grace period of thirty days is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits.",
    "There is a waiting period of thirty-six (36) months of continuous coverage from the first policy inception for pre-existing diseases and their direct complications to be covered.",
    "Yes, the policy covers maternity expenses, including childbirth and lawful medical termination of pregnancy. To be eligible, the female insured person must have been continuously covered for at least 24 months. The benefit is limited to two deliveries or terminations during the policy period.",
    "The policy has a specific waiting period of two (2) years for cataract surgery.",
    "Yes, the policy indemnifies the medical expenses for the organ donor's hospitalization for the purpose of harvesting the organ, provided the organ is for an insured person and the donation complies with the Transplantation of Human Organs Act, 1994.",
    "A No Claim Discount of 5% on the base premium is offered on renewal for a one-year policy term if no claims were made in the preceding year. The maximum aggregate NCD is capped at 5% of the total base premium.",
    "Yes, the policy reimburses expenses for health check-ups at the end of every block of two continuous policy years, provided the policy has been renewed without a break. The amount is subject to the limits specified in the Table of Benefits.",
    "A hospital is defined as an institution with at least 10 inpatient beds (in towns with a population below ten lakhs) or 15 beds (in all other places), with qualified nursing staff and medical practitioners available 24/7, a fully equipped operation theatre, and which maintains daily records of patients.",
    "The policy covers medical expenses for inpatient treatment under Ayurveda, Yoga, Naturopathy, Unani, Siddha, and Homeopathy systems up to the Sum Insured limit, provided the treatment is taken in an AYUSH Hospital.",
    "Yes, for Plan A, the daily room rent is capped at 1% of the Sum Insured, and ICU charges are capped at 2% of the Sum Insured. These limits do not apply if the treatment is for a listed procedure in a Preferred Provider Network (PPN)."
]

class HackRxTester:
    """Complete testing suite for HackRx 6.0 system"""
    
    def __init__(self, base_url: str, auth_token: str):
        self.base_url = base_url.rstrip('/')
        self.auth_token = auth_token
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {auth_token}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        })
    
    def test_health_endpoint(self) -> bool:
        """Test the health check endpoint"""
        print("🏥 Testing health endpoint...")
        
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=30)
            
            if response.status_code == 200:
                health_data = response.json()
                print("✅ Health check passed")
                print(f"   Status: {health_data.get('status', 'unknown')}")
                
                components = health_data.get('components', {})
                for component, status in components.items():
                    print(f"   {component}: {status}")
                
                return True
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Health check error: {e}")
            return False
    
    def test_root_endpoint(self) -> bool:
        """Test the root endpoint"""
        print("🏠 Testing root endpoint...")
        
        try:
            response = self.session.get(f"{self.base_url}/", timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                print("✅ Root endpoint working")
                print(f"   System: {data.get('system', 'unknown')}")
                print(f"   Version: {data.get('version', 'unknown')}")
                return True
            else:
                print(f"❌ Root endpoint failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Root endpoint error: {e}")
            return False
    
    def test_authentication(self) -> bool:
        """Test authentication with wrong token"""
        print("🔐 Testing authentication...")
        
        try:
            # Test with wrong token
            wrong_session = requests.Session()
            wrong_session.headers.update({
                "Authorization": "Bearer wrong-token",
                "Content-Type": "application/json"
            })
            
            response = wrong_session.post(
                f"{self.base_url}/hackrx/run",
                json={
                    "documents": "https://example.com/test.pdf",
                    "questions": ["test question"]
                },
                timeout=30
            )
            
            if response.status_code == 401:
                print("✅ Authentication properly enforced")
                return True
            else:
                print(f"❌ Authentication not working: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Authentication test error: {e}")
            return False
    
    def test_hackrx_endpoint_basic(self) -> bool:
        """Test basic HackRx endpoint functionality"""
        print("🎯 Testing HackRx endpoint (basic)...")
        
        try:
            # Simple test with minimal data
            test_payload = {
                "documents": TEST_DOCUMENT_URL,
                "questions": ["What is this policy about?"]
            }
            
            start_time = time.time()
            response = self.session.post(
                f"{self.base_url}/hackrx/run",
                json=test_payload,
                timeout=120  # 2 minutes timeout
            )
            end_time = time.time()
            
            response_time = end_time - start_time
            
            if response.status_code == 200:
                data = response.json()
                answers = data.get('answers', [])
                
                print("✅ HackRx endpoint working")
                print(f"   Response time: {response_time:.2f} seconds")
                print(f"   Answers received: {len(answers)}")
                
                if answers:
                    print(f"   Sample answer: {answers[0][:100]}...")
                
                return True
            else:
                print(f"❌ HackRx endpoint failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ HackRx endpoint error: {e}")
            return False
    
    def test_hackrx_endpoint_full(self) -> Dict[str, any]:
        """Test HackRx endpoint with full test data"""
        print("🏆 Testing HackRx endpoint (full test suite)...")
        
        try:
            test_payload = {
                "documents": TEST_DOCUMENT_URL,
                "questions": TEST_QUESTIONS
            }
            
            print(f"📊 Testing with {len(TEST_QUESTIONS)} questions...")
            
            start_time = time.time()
            response = self.session.post(
                f"{self.base_url}/hackrx/run",
                json=test_payload,
                timeout=300  # 5 minutes timeout for full test
            )
            end_time = time.time()
            
            response_time = end_time - start_time
            
            if response.status_code == 200:
                data = response.json()
                answers = data.get('answers', [])
                
                print("✅ Full HackRx test completed")
                print(f"   Response time: {response_time:.2f} seconds")
                print(f"   Questions processed: {len(TEST_QUESTIONS)}")
                print(f"   Answers received: {len(answers)}")
                print(f"   Average time per question: {response_time/len(TEST_QUESTIONS):.2f}s")
                
                # Calculate accuracy metrics
                accuracy_results = self.calculate_accuracy(answers, EXPECTED_ANSWERS)
                
                return {
                    "success": True,
                    "response_time": response_time,
                    "questions_count": len(TEST_QUESTIONS),
                    "answers_count": len(answers),
                    "answers": answers,
                    "accuracy": accuracy_results
                }
            else:
                print(f"❌ Full HackRx test failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return {"success": False, "error": response.text}
                
        except Exception as e:
            print(f"❌ Full HackRx test error: {e}")
            return {"success": False, "error": str(e)}
    
    def calculate_accuracy(self, received_answers: List[str], expected_answers: List[str]) -> Dict[str, any]:
        """Calculate accuracy metrics"""
        print("\n📊 Analyzing answer accuracy...")
        
        if len(received_answers) != len(expected_answers):
            print(f"⚠️ Answer count mismatch: {len(received_answers)} vs {len(expected_answers)}")
        
        accuracy_scores = []
        keyword_matches = []
        
        for i, (received, expected) in enumerate(zip(received_answers, expected_answers)):
            # Simple keyword matching
            expected_keywords = set(expected.lower().split())
            received_keywords = set(received.lower().split())
            
            common_keywords = expected_keywords.intersection(received_keywords)
            keyword_match_ratio = len(common_keywords) / len(expected_keywords) if expected_keywords else 0
            
            keyword_matches.append(keyword_match_ratio)
            
            # Length similarity
            length_ratio = min(len(received), len(expected)) / max(len(received), len(expected))
            
            # Combined score
            combined_score = (keyword_match_ratio * 0.7) + (length_ratio * 0.3)
            accuracy_scores.append(combined_score)
            
            print(f"   Q{i+1}: Keyword match: {keyword_match_ratio:.2f}, Combined score: {combined_score:.2f}")
        
        avg_accuracy = sum(accuracy_scores) / len(accuracy_scores) if accuracy_scores else 0
        avg_keyword_match = sum(keyword_matches) / len(keyword_matches) if keyword_matches else 0
        
        print(f"\n📈 Overall Accuracy Metrics:")
        print(f"   Average accuracy score: {avg_accuracy:.2f}")
        print(f"   Average keyword match: {avg_keyword_match:.2f}")
        
        return {
            "average_accuracy": avg_accuracy,
            "average_keyword_match": avg_keyword_match,
            "individual_scores": accuracy_scores,
            "keyword_matches": keyword_matches
        }
    
    def test_error_handling(self) -> bool:
        """Test error handling with invalid inputs"""
        print("🔍 Testing error handling...")
        
        test_cases = [
            {
                "name": "Invalid document URL",
                "payload": {
                    "documents": "https://invalid-url.com/nonexistent.pdf",
                    "questions": ["test question"]
                },
                "expected_status": [400, 500]
            },
            {
                "name": "Empty questions",
                "payload": {
                    "documents": TEST_DOCUMENT_URL,
                    "questions": []
                },
                "expected_status": [400, 422]
            },
            {
                "name": "Invalid JSON",
                "payload": "invalid json",
                "expected_status": [400, 422]
            }
        ]
        
        all_passed = True
        
        for test_case in test_cases:
            try:
                print(f"   Testing: {test_case['name']}")
                
                if isinstance(test_case['payload'], str):
                    # Invalid JSON test
                    response = requests.post(
                        f"{self.base_url}/hackrx/run",
                        data=test_case['payload'],
                        headers={"Authorization": f"Bearer {self.auth_token}"},
                        timeout=30
                    )
                else:
                    response = self.session.post(
                        f"{self.base_url}/hackrx/run",
                        json=test_case['payload'],
                        timeout=30
                    )
                
                if response.status_code in test_case['expected_status']:
                    print(f"   ✅ {test_case['name']}: Properly handled")
                else:
                    print(f"   ❌ {test_case['name']}: Unexpected status {response.status_code}")
                    all_passed = False
                    
            except Exception as e:
                print(f"   ❌ {test_case['name']}: Error {e}")
                all_passed = False
        
        return all_passed
    
    def run_performance_test(self, num_iterations: int = 3) -> Dict[str, any]:
        """Run performance tests"""
        print(f"⚡ Running performance test ({num_iterations} iterations)...")
        
        response_times = []
        success_count = 0
        
        test_payload = {
            "documents": TEST_DOCUMENT_URL,
            "questions": TEST_QUESTIONS[:3]  # Use fewer questions for performance test
        }
        
        for i in range(num_iterations):
            try:
                print(f"   Iteration {i+1}/{num_iterations}")
                
                start_time = time.time()
                response = self.session.post(
                    f"{self.base_url}/hackrx/run",
                    json=test_payload,
                    timeout=120
                )
                end_time = time.time()
                
                response_time = end_time - start_time
                response_times.append(response_time)
                
                if response.status_code == 200:
                    success_count += 1
                    print(f"   ✅ Iteration {i+1}: {response_time:.2f}s")
                else:
                    print(f"   ❌ Iteration {i+1}: Failed ({response.status_code})")
                    
            except Exception as e:
                print(f"   ❌ Iteration {i+1}: Error {e}")
        
        if response_times:
            avg_time = sum(response_times) / len(response_times)
            min_time = min(response_times)
            max_time = max(response_times)
            
            print(f"\n⚡ Performance Results:")
            print(f"   Success rate: {success_count}/{num_iterations} ({success_count/num_iterations*100:.1f}%)")
            print(f"   Average response time: {avg_time:.2f}s")
            print(f"   Fastest response: {min_time:.2f}s")
            print(f"   Slowest response: {max_time:.2f}s")
            
            return {
                "success_rate": success_count / num_iterations,
                "average_time": avg_time,
                "min_time": min_time,
                "max_time": max_time,
                "all_times": response_times
            }
        else:
            return {"success_rate": 0, "error": "No successful responses"}

def main():
    """Run complete test suite"""
    print("🏆 HackRx 6.0 - Complete Testing Suite")
    print("=" * 60)
    print(f"🎯 Testing endpoint: {BASE_URL}")
    print("=" * 60)
    
    tester = HackRxTester(BASE_URL, AUTH_TOKEN)
    
    # Run all tests
    tests_passed = 0
    total_tests = 6
    
    # 1. Health check
    if tester.test_health_endpoint():
        tests_passed += 1
    
    # 2. Root endpoint  
    if tester.test_root_endpoint():
        tests_passed += 1
    
    # 3. Authentication
    if tester.test_authentication():
        tests_passed += 1
    
    # 4. Basic HackRx test
    if tester.test_hackrx_endpoint_basic():
        tests_passed += 1
    
    # 5. Error handling
    if tester.test_error_handling():
        tests_passed += 1
    
    # 6. Full HackRx test
    print("\n" + "="*60)
    full_test_results = tester.test_hackrx_endpoint_full()
    if full_test_results.get("success"):
        tests_passed += 1
    
    # Performance test
    print("\n" + "="*60)
    performance_results = tester.run_performance_test(3)
    
    # Final summary
    print("\n" + "="*60)
    print("🏆 FINAL TEST RESULTS")
    print("="*60)
    print(f"✅ Tests passed: {tests_passed}/{total_tests}")
    print(f"📊 Success rate: {tests_passed/total_tests*100:.1f}%")
    
    if full_test_results.get("success"):
        accuracy = full_test_results["accuracy"]
        print(f"🎯 Answer accuracy: {accuracy['average_accuracy']:.2f}")
        print(f"⚡ Average response time: {full_test_results['response_time']:.2f}s")
    
    if performance_results.get("success_rate"):
        print(f"🚀 Performance success rate: {performance_results['success_rate']*100:.1f}%")
        print(f"⏱️ Average performance time: {performance_results['average_time']:.2f}s")
    
    # Deployment readiness
    if tests_passed >= 5:
        print("\n🎉 SYSTEM READY FOR HACKRX SUBMISSION!")
        print("📝 Next steps:")
        print("   1. Deploy to production platform")
        print("   2. Update BASE_URL in this script to your deployed URL")
        print("   3. Run tests again with production URL")
        print("   4. Submit webhook URL to HackRx platform")
    else:
        print("\n⚠️ SYSTEM NEEDS FIXES BEFORE SUBMISSION")
        print("❌ Fix failing tests before deploying")
    
    print("="*60)

if __name__ == "__main__":
    main()