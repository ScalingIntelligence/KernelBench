#!/usr/bin/env python3
"""
Comprehensive unit tests for AIDE + KernelBench integration
Tests all edge cases, especially empty journals and error handling
"""

import sys
import tempfile
from pathlib import Path
import torch
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

import unittest
from unittest.mock import Mock, patch, MagicMock

# Import components to test
from journal import Journal, Node
from interpreter import ExecutionResult
from utils.metric import MetricValue, WorstMetricValue
from utils.tree_export import (
    generate_layout,
    normalize_layout,
    cfg_to_tree_struct,
    get_edges,
)
from utils.config import save_run
from utils.serialize import dumps_json, dump_json, load_json
from omegaconf import OmegaConf


class TestJournal(unittest.TestCase):
    """Test Journal class, especially edge cases with empty journals"""
    
    def test_empty_journal(self):
        """Test empty journal initialization"""
        journal = Journal()
        self.assertEqual(len(journal), 0)
        self.assertEqual(journal.nodes, [])
        self.assertEqual(journal.draft_nodes, [])
        self.assertEqual(journal.buggy_nodes, [])
        self.assertEqual(journal.good_nodes, [])
    
    def test_empty_journal_get_best_node(self):
        """Test get_best_node with empty journal"""
        journal = Journal()
        self.assertIsNone(journal.get_best_node(only_good=True))
        self.assertIsNone(journal.get_best_node(only_good=False))
    
    def test_empty_journal_get_metric_history(self):
        """Test get_metric_history with empty journal"""
        journal = Journal()
        self.assertEqual(journal.get_metric_history(), [])
    
    def test_single_node_journal(self):
        """Test journal with one node"""
        journal = Journal()
        node = Node(code="test", plan="test plan")
        node.metric = MetricValue(1.5, maximize=False)
        node.is_buggy = False
        journal.append(node)
        
        self.assertEqual(len(journal), 1)
        self.assertEqual(len(journal.draft_nodes), 1)
        self.assertEqual(len(journal.good_nodes), 1)
        self.assertEqual(len(journal.buggy_nodes), 0)
        self.assertEqual(journal.get_best_node(), node)
    
    def test_buggy_node_journal(self):
        """Test journal with only buggy nodes"""
        journal = Journal()
        node = Node(code="test", plan="test plan")
        node.metric = WorstMetricValue()
        node.is_buggy = True
        journal.append(node)
        
        self.assertEqual(len(journal), 1)
        self.assertEqual(len(journal.buggy_nodes), 1)
        self.assertEqual(len(journal.good_nodes), 0)
        self.assertIsNone(journal.get_best_node(only_good=True))
        self.assertEqual(journal.get_best_node(only_good=False), node)
    
    def test_multiple_nodes_best_selection(self):
        """Test best node selection with multiple nodes"""
        journal = Journal()
        
        # Create nodes with different metrics
        node1 = Node(code="test1", plan="plan1")
        node1.metric = MetricValue(2.0, maximize=False)  # worse
        node1.is_buggy = False
        journal.append(node1)
        
        node2 = Node(code="test2", plan="plan2")
        node2.metric = MetricValue(1.0, maximize=False)  # better (lower)
        node2.is_buggy = False
        journal.append(node2)
        
        node3 = Node(code="test3", plan="plan3")
        node3.metric = MetricValue(1.5, maximize=False)  # middle
        node3.is_buggy = False
        journal.append(node3)
        
        best = journal.get_best_node()
        self.assertEqual(best, node2)
        self.assertEqual(best.metric.value, 1.0)
    
    def test_node_parent_child_relationship(self):
        """Test parent-child relationships in nodes"""
        journal = Journal()
        
        parent = Node(code="parent", plan="parent plan")
        parent.is_buggy = False
        journal.append(parent)
        
        child = Node(code="child", plan="child plan", parent=parent)
        child.is_buggy = False
        journal.append(child)
        
        self.assertIn(child, parent.children)
        self.assertEqual(child.parent, parent)
        self.assertEqual(len(journal.draft_nodes), 1)  # Only parent is draft


class TestTreeExport(unittest.TestCase):
    """Test tree export and visualization functions"""
    
    def test_generate_layout_empty(self):
        """Test generate_layout with empty graph"""
        layout = generate_layout(0, [])
        self.assertEqual(len(layout), 0)
    
    def test_generate_layout_single_node(self):
        """Test generate_layout with single node"""
        layout = generate_layout(1, [])
        self.assertEqual(len(layout), 1)
        self.assertEqual(layout.shape, (1, 2))
    
    def test_generate_layout_multiple_nodes(self):
        """Test generate_layout with multiple nodes"""
        edges = [(0, 1), (0, 2), (1, 3)]
        layout = generate_layout(4, edges)
        self.assertEqual(len(layout), 4)
        self.assertEqual(layout.shape, (4, 2))
    
    def test_normalize_layout_empty(self):
        """Test normalize_layout with empty array"""
        layout = np.array([])
        normalized = normalize_layout(layout)
        self.assertEqual(len(normalized), 0)
    
    def test_normalize_layout_single_point(self):
        """Test normalize_layout with single point"""
        layout = np.array([[1.0, 2.0]])
        normalized = normalize_layout(layout)
        self.assertEqual(normalized.shape, (1, 2))
        # Single point should be normalized to (0.5, 0) due to nan handling
    
    def test_cfg_to_tree_struct_empty_journal(self):
        """Test cfg_to_tree_struct with empty journal"""
        cfg = OmegaConf.create({"exp_name": "test"})
        journal = Journal()
        
        struct = cfg_to_tree_struct(cfg, journal)
        
        self.assertEqual(struct["edges"], [])
        self.assertEqual(struct["layout"], [])
        self.assertEqual(struct["plan"], [])
        self.assertEqual(struct["code"], [])
        self.assertEqual(struct["term_out"], [])
        self.assertEqual(struct["analysis"], [])
        self.assertEqual(struct["exp_name"], "test")
        self.assertEqual(struct["metrics"], [])
    
    def test_cfg_to_tree_struct_with_nodes(self):
        """Test cfg_to_tree_struct with nodes"""
        cfg = OmegaConf.create({"exp_name": "test"})
        journal = Journal()
        
        node = Node(code="test code", plan="test plan")
        node.is_buggy = False
        node.analysis = "test analysis"
        journal.append(node)
        
        struct = cfg_to_tree_struct(cfg, journal)
        
        self.assertEqual(len(struct["edges"]), 0)  # No edges for single node
        self.assertEqual(len(struct["layout"]), 1)
        self.assertEqual(len(struct["plan"]), 1)
        self.assertEqual(len(struct["code"]), 1)
        self.assertEqual(len(struct["metrics"]), 1)
    
    def test_get_edges_empty(self):
        """Test get_edges with empty journal"""
        journal = Journal()
        edges = list(get_edges(journal))
        self.assertEqual(edges, [])
    
    def test_get_edges_with_parent_child(self):
        """Test get_edges with parent-child relationship"""
        journal = Journal()
        
        parent = Node(code="parent", plan="parent")
        journal.append(parent)
        
        child = Node(code="child", plan="child", parent=parent)
        journal.append(child)
        
        edges = list(get_edges(journal))
        self.assertEqual(edges, [(0, 1)])


class TestSerialization(unittest.TestCase):
    """Test JSON serialization and deserialization"""
    
    def test_serialize_empty_journal(self):
        """Test serializing empty journal"""
        journal = Journal()
        json_str = dumps_json(journal)
        self.assertIsInstance(json_str, str)
        self.assertIn('"nodes":[]', json_str)
    
    def test_serialize_journal_with_nodes(self):
        """Test serializing journal with nodes"""
        journal = Journal()
        
        node = Node(code="test", plan="plan")
        node.is_buggy = False
        node.metric = MetricValue(1.5, maximize=False)
        journal.append(node)
        
        json_str = dumps_json(journal)
        self.assertIsInstance(json_str, str)
        self.assertIn('"code":"test"', json_str)
    
    def test_serialize_deserialize_roundtrip(self):
        """Test serialization roundtrip"""
        journal = Journal()
        
        node1 = Node(code="test1", plan="plan1")
        node1.is_buggy = False
        journal.append(node1)
        
        node2 = Node(code="test2", plan="plan2", parent=node1)
        node2.is_buggy = True
        journal.append(node2)
        
        # Serialize
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            dump_json(journal, Path(f.name))
            temp_path = f.name
        
        try:
            # Deserialize
            loaded_journal = load_json(Path(temp_path), Journal)
            
            self.assertEqual(len(loaded_journal), len(journal))
            self.assertEqual(loaded_journal[0].code, journal[0].code)
            self.assertEqual(loaded_journal[1].code, journal[1].code)
        finally:
            Path(temp_path).unlink()


class TestSaveRun(unittest.TestCase):
    """Test save_run function with edge cases"""
    
    def test_save_run_empty_journal(self):
        """Test save_run with empty journal"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = OmegaConf.create({
                "log_dir": tmpdir,
                "exp_name": "test",
            })
            journal = Journal()
            
            # Should not crash
            save_run(cfg, journal)
            
            # Check files created
            log_dir = Path(tmpdir)
            self.assertTrue((log_dir / "journal.json").exists())
            self.assertTrue((log_dir / "config.yaml").exists())
            # No tree_plot.html for empty journal
            self.assertFalse((log_dir / "tree_plot.html").exists())
            # No best_solution.py for empty journal
            self.assertFalse((log_dir / "best_solution.py").exists())
    
    def test_save_run_with_nodes(self):
        """Test save_run with nodes"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = OmegaConf.create({
                "log_dir": tmpdir,
                "exp_name": "test",
            })
            journal = Journal()
            
            node = Node(code="test code", plan="test plan")
            node.is_buggy = False
            node.metric = MetricValue(1.0, maximize=False)
            journal.append(node)
            
            save_run(cfg, journal)
            
            log_dir = Path(tmpdir)
            self.assertTrue((log_dir / "journal.json").exists())
            self.assertTrue((log_dir / "config.yaml").exists())
            self.assertTrue((log_dir / "tree_plot.html").exists())
            self.assertTrue((log_dir / "best_solution.py").exists())
            
            # Check best solution content
            with open(log_dir / "best_solution.py") as f:
                content = f.read()
                self.assertEqual(content, "test code")


class TestNode(unittest.TestCase):
    """Test Node class"""
    
    def test_node_creation(self):
        """Test basic node creation"""
        node = Node(code="test", plan="plan")
        self.assertEqual(node.code, "test")
        self.assertEqual(node.plan, "plan")
        self.assertIsNone(node.parent)
        self.assertEqual(len(node.children), 0)
    
    def test_node_stage_name(self):
        """Test node stage name determination"""
        # Draft node (no parent)
        draft = Node(code="draft", plan="draft plan")
        self.assertEqual(draft.stage_name, "draft")
        
        # Debug node (parent is buggy)
        parent_buggy = Node(code="parent", plan="parent plan")
        parent_buggy.is_buggy = True
        debug = Node(code="debug", plan="debug plan", parent=parent_buggy)
        self.assertEqual(debug.stage_name, "debug")
        
        # Improve node (parent is good)
        parent_good = Node(code="parent", plan="parent plan")
        parent_good.is_buggy = False
        improve = Node(code="improve", plan="improve plan", parent=parent_good)
        self.assertEqual(improve.stage_name, "improve")
    
    def test_node_is_leaf(self):
        """Test is_leaf property"""
        parent = Node(code="parent", plan="parent")
        self.assertTrue(parent.is_leaf)
        
        child = Node(code="child", plan="child", parent=parent)
        self.assertFalse(parent.is_leaf)
        self.assertTrue(child.is_leaf)
    
    def test_node_debug_depth(self):
        """Test debug_depth calculation"""
        # Non-debug node
        draft = Node(code="draft", plan="draft")
        draft.is_buggy = False
        self.assertEqual(draft.debug_depth, 0)
        
        # First debug
        buggy1 = Node(code="buggy1", plan="buggy1")
        buggy1.is_buggy = True
        debug1 = Node(code="debug1", plan="debug1", parent=buggy1)
        self.assertEqual(debug1.debug_depth, 1)
        
        # Second debug
        debug1.is_buggy = True
        debug2 = Node(code="debug2", plan="debug2", parent=debug1)
        self.assertEqual(debug2.debug_depth, 2)
    
    def test_node_absorb_exec_result(self):
        """Test absorbing execution results"""
        node = Node(code="test", plan="plan")
        
        exec_result = ExecutionResult(
            term_out=["output line 1", "output line 2"],
            exec_time=1.5,
            exc_type="TestError",
            exc_info={"error": "test"},
            exc_stack=None,
        )
        
        node.absorb_exec_result(exec_result)
        
        self.assertEqual(node.exec_time, 1.5)
        self.assertEqual(node.exc_type, "TestError")
        self.assertIsNotNone(node.term_out)


class TestMetrics(unittest.TestCase):
    """Test metric handling"""
    
    def test_metric_value_comparison(self):
        """Test metric value comparison"""
        m1 = MetricValue(1.0, maximize=False)
        m2 = MetricValue(2.0, maximize=False)
        m3 = MetricValue(1.5, maximize=False)
        
        # Lower is better for minimize
        self.assertGreater(m1, m2)  # 1.0 > 2.0 (better)
        self.assertGreater(m1, m3)
        self.assertGreater(m3, m2)
    
    def test_worst_metric_value(self):
        """Test worst metric value"""
        worst = WorstMetricValue()
        normal = MetricValue(100.0, maximize=False)
        
        self.assertLess(worst, normal)  # Worst is always worse


def run_tests(verbose=True):
    """Run all tests and return results"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestJournal))
    suite.addTests(loader.loadTestsFromTestCase(TestTreeExport))
    suite.addTests(loader.loadTestsFromTestCase(TestSerialization))
    suite.addTests(loader.loadTestsFromTestCase(TestSaveRun))
    suite.addTests(loader.loadTestsFromTestCase(TestNode))
    suite.addTests(loader.loadTestsFromTestCase(TestMetrics))
    
    runner = unittest.TextTestRunner(verbosity=2 if verbose else 1)
    result = runner.run(suite)
    
    return result


if __name__ == "__main__":
    print("="*80)
    print("AIDE + KernelBench Integration - Comprehensive Unit Tests")
    print("="*80)
    print()
    
    result = run_tests(verbose=True)
    
    print()
    print("="*80)
    print("Test Summary")
    print("="*80)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print()
        print("✅ All tests passed!")
        sys.exit(0)
    else:
        print()
        print("❌ Some tests failed")
        sys.exit(1)
