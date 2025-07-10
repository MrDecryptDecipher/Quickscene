#!/usr/bin/env python3
"""
Performance Benchmarking Tool for Quickscene

Comprehensive performance testing and reporting for the video search system.
Tests query response times, accuracy, and system performance under load.
"""

import time
import json
import statistics
from pathlib import Path
from typing import Dict, List, Any, Tuple
import logging
from datetime import datetime
import concurrent.futures
import sys

# Add app to path
sys.path.insert(0, str(Path(__file__).parent))

from app.config import get_config
from app.production_query_handler import ProductionQueryHandler

class QuicksceneBenchmark:
    """Comprehensive benchmarking suite for Quickscene system"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize benchmark suite"""
        self.config = get_config(config_path)
        self.logger = logging.getLogger(__name__)
        self.query_handler = None
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'system_info': {},
            'performance_tests': {},
            'load_tests': {},
            'accuracy_tests': {},
            'summary': {}
        }
    
    def _initialize_system(self):
        """Initialize the query handler"""
        try:
            self.query_handler = ProductionQueryHandler()
            self.logger.info("Query handler initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize query handler: {e}")
            return False
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Collect system information"""
        import platform
        import psutil
        
        return {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'memory_gb': round(psutil.virtual_memory().total / (1024**3), 2),
            'cpu_freq_mhz': psutil.cpu_freq().current if psutil.cpu_freq() else 'Unknown'
        }
    
    def test_single_query_performance(self) -> Dict[str, Any]:
        """Test single query performance across different query types"""
        test_queries = [
            # Keyword searches (should be fastest)
            ("AI", "keyword"),
            ("blockchain", "keyword"),
            ("finance", "keyword"),
            ("quantum", "keyword"),
            ("machine", "keyword"),
            
            # Semantic searches (slightly slower)
            ("artificial intelligence", "semantic"),
            ("machine learning algorithms", "semantic"),
            ("quantum computing principles", "semantic"),
            ("blockchain technology", "semantic"),
            ("financial markets", "semantic"),
            
            # Complex semantic searches
            ("how does artificial intelligence work", "semantic"),
            ("explain quantum computing concepts", "semantic"),
            ("what is machine learning", "semantic")
        ]
        
        results = {
            'keyword_queries': [],
            'semantic_queries': [],
            'complex_queries': [],
            'all_queries': []
        }
        
        self.logger.info("Running single query performance tests...")
        
        for query, query_type in test_queries:
            # Warm up
            self.query_handler.query(query, top_k=5)
            
            # Measure performance over multiple runs
            times = []
            for _ in range(10):  # 10 runs per query
                start_time = time.time()
                result = self.query_handler.query(query, top_k=5)
                end_time = time.time()
                
                query_time_ms = (end_time - start_time) * 1000
                times.append(query_time_ms)
            
            avg_time = statistics.mean(times)
            min_time = min(times)
            max_time = max(times)
            std_dev = statistics.stdev(times) if len(times) > 1 else 0
            
            query_result = {
                'query': query,
                'type': query_type,
                'avg_time_ms': round(avg_time, 2),
                'min_time_ms': round(min_time, 2),
                'max_time_ms': round(max_time, 2),
                'std_dev_ms': round(std_dev, 2),
                'meets_requirement': avg_time < 700,  # <700ms requirement
                'results_count': len(result.get('results', []))
            }
            
            results['all_queries'].append(query_result)
            
            if query_type == "keyword":
                results['keyword_queries'].append(query_result)
            elif len(query.split()) <= 3:
                results['semantic_queries'].append(query_result)
            else:
                results['complex_queries'].append(query_result)
        
        # Calculate category averages
        for category in ['keyword_queries', 'semantic_queries', 'complex_queries']:
            if results[category]:
                avg_time = statistics.mean([q['avg_time_ms'] for q in results[category]])
                results[f'{category}_avg_ms'] = round(avg_time, 2)
        
        return results
    
    def test_load_performance(self) -> Dict[str, Any]:
        """Test system performance under concurrent load"""
        test_queries = [
            "artificial intelligence",
            "blockchain",
            "quantum computing",
            "machine learning",
            "finance"
        ]
        
        results = {
            'concurrent_users': [],
            'throughput_tests': []
        }
        
        self.logger.info("Running load performance tests...")
        
        # Test different concurrency levels
        for concurrent_users in [1, 5, 10, 20]:
            self.logger.info(f"Testing with {concurrent_users} concurrent users...")
            
            def run_query(query):
                start_time = time.time()
                result = self.query_handler.query(query, top_k=5)
                end_time = time.time()
                return (end_time - start_time) * 1000, len(result.get('results', []))
            
            # Run concurrent queries
            start_time = time.time()
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_users) as executor:
                futures = []
                for _ in range(concurrent_users * 5):  # 5 queries per user
                    query = test_queries[_ % len(test_queries)]
                    futures.append(executor.submit(run_query, query))
                
                query_times = []
                result_counts = []
                for future in concurrent.futures.as_completed(futures):
                    query_time, result_count = future.result()
                    query_times.append(query_time)
                    result_counts.append(result_count)
            
            total_time = time.time() - start_time
            
            load_result = {
                'concurrent_users': concurrent_users,
                'total_queries': len(query_times),
                'total_time_seconds': round(total_time, 2),
                'queries_per_second': round(len(query_times) / total_time, 2),
                'avg_query_time_ms': round(statistics.mean(query_times), 2),
                'max_query_time_ms': round(max(query_times), 2),
                'min_query_time_ms': round(min(query_times), 2),
                'avg_results_per_query': round(statistics.mean(result_counts), 2)
            }
            
            results['concurrent_users'].append(load_result)
        
        return results
    
    def test_accuracy(self) -> Dict[str, Any]:
        """Test search accuracy with known queries"""
        accuracy_tests = [
            {
                'query': 'artificial intelligence',
                'expected_video': 'What is Artificial Superintelligence (ASI)_',
                'description': 'Should find AI-related content'
            },
            {
                'query': 'blockchain',
                'expected_video': 'Hyperledger Besu Explained',
                'description': 'Should find blockchain-related content'
            },
            {
                'query': 'quantum computing',
                'expected_video': "A beginner's guide to quantum computing _ Shohini Ghose",
                'description': 'Should find quantum computing content'
            },
            {
                'query': 'finance',
                'expected_video': 'What is Artificial Superintelligence (ASI)_',
                'description': 'Should find finance mentions'
            }
        ]
        
        results = {
            'accuracy_tests': [],
            'overall_accuracy': 0
        }
        
        self.logger.info("Running accuracy tests...")
        
        correct_predictions = 0
        
        for test in accuracy_tests:
            result = self.query_handler.query(test['query'], top_k=5)
            
            # Check if expected video is in top results
            found_expected = False
            rank = None
            
            for i, res in enumerate(result.get('results', [])):
                if test['expected_video'] in res.get('video_id', ''):
                    found_expected = True
                    rank = i + 1
                    break
            
            if found_expected:
                correct_predictions += 1
            
            accuracy_result = {
                'query': test['query'],
                'expected_video': test['expected_video'],
                'description': test['description'],
                'found_expected': found_expected,
                'rank': rank,
                'total_results': len(result.get('results', [])),
                'top_result': result.get('results', [{}])[0].get('video_id', 'No results') if result.get('results') else 'No results'
            }
            
            results['accuracy_tests'].append(accuracy_result)
        
        results['overall_accuracy'] = round((correct_predictions / len(accuracy_tests)) * 100, 1)
        
        return results
    
    def run_full_benchmark(self) -> Dict[str, Any]:
        """Run complete benchmark suite"""
        self.logger.info("Starting comprehensive benchmark suite...")
        
        # Initialize system
        if not self._initialize_system():
            return {'error': 'Failed to initialize system'}
        
        # Collect system info
        self.results['system_info'] = self._get_system_info()
        
        # Run performance tests
        self.results['performance_tests'] = self.test_single_query_performance()
        
        # Run load tests
        self.results['load_tests'] = self.test_load_performance()
        
        # Run accuracy tests
        self.results['accuracy_tests'] = self.test_accuracy()
        
        # Generate summary
        self._generate_summary()
        
        return self.results
    
    def _generate_summary(self):
        """Generate benchmark summary"""
        perf = self.results['performance_tests']
        load = self.results['load_tests']
        accuracy = self.results['accuracy_tests']
        
        # Performance summary
        all_queries = perf.get('all_queries', [])
        if all_queries:
            avg_response_time = statistics.mean([q['avg_time_ms'] for q in all_queries])
            fastest_query = min(all_queries, key=lambda x: x['avg_time_ms'])
            slowest_query = max(all_queries, key=lambda x: x['avg_time_ms'])
            
            queries_meeting_requirement = sum(1 for q in all_queries if q['meets_requirement'])
            requirement_compliance = (queries_meeting_requirement / len(all_queries)) * 100
        else:
            avg_response_time = 0
            fastest_query = {}
            slowest_query = {}
            requirement_compliance = 0
        
        # Load summary
        max_throughput = max([test['queries_per_second'] for test in load.get('concurrent_users', [])], default=0)
        
        self.results['summary'] = {
            'avg_response_time_ms': round(avg_response_time, 2),
            'fastest_query_ms': fastest_query.get('avg_time_ms', 0),
            'slowest_query_ms': slowest_query.get('avg_time_ms', 0),
            'requirement_compliance_percent': round(requirement_compliance, 1),
            'max_throughput_qps': round(max_throughput, 2),
            'search_accuracy_percent': accuracy.get('overall_accuracy', 0),
            'total_queries_tested': len(all_queries),
            'system_status': 'PASS' if requirement_compliance > 90 and accuracy.get('overall_accuracy', 0) > 75 else 'FAIL'
        }
    
    def save_report(self, output_path: str = "benchmark_report.json"):
        """Save benchmark report to file"""
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        self.logger.info(f"Benchmark report saved to {output_path}")
    
    def print_summary(self):
        """Print benchmark summary to console"""
        summary = self.results.get('summary', {})
        
        print("\n" + "="*60)
        print("🚀 QUICKSCENE PERFORMANCE BENCHMARK REPORT")
        print("="*60)
        
        print(f"📊 Average Response Time: {summary.get('avg_response_time_ms', 0):.2f}ms")
        print(f"⚡ Fastest Query: {summary.get('fastest_query_ms', 0):.2f}ms")
        print(f"🐌 Slowest Query: {summary.get('slowest_query_ms', 0):.2f}ms")
        print(f"✅ Requirement Compliance: {summary.get('requirement_compliance_percent', 0):.1f}%")
        print(f"🔥 Max Throughput: {summary.get('max_throughput_qps', 0):.2f} queries/sec")
        print(f"🎯 Search Accuracy: {summary.get('search_accuracy_percent', 0):.1f}%")
        print(f"📈 System Status: {summary.get('system_status', 'UNKNOWN')}")
        
        print("\n" + "="*60)

def main():
    """CLI interface for benchmarking"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Quickscene Performance Benchmark")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--output", default="benchmark_report.json", help="Output report file")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark (fewer iterations)")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Run benchmark
    benchmark = QuicksceneBenchmark(args.config)
    
    try:
        results = benchmark.run_full_benchmark()
        
        if 'error' in results:
            print(f"❌ Benchmark failed: {results['error']}")
            return 1
        
        # Save and display results
        benchmark.save_report(args.output)
        benchmark.print_summary()
        
        return 0
        
    except Exception as e:
        print(f"❌ Benchmark error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
