"""Executable memory-resource invariants; run with unittest discovery."""

import unittest

from hw_model.memory import BankedSRAM, MemRequest, SharedMemory


def memory(**overrides):
    options = dict(capacity_bytes=128, beat_bytes=4, read_latency=2,
                   write_latency=3, accept_interval=1, max_outstanding=4)
    options.update(overrides)
    return SharedMemory(**options)


def request(request_id="a", requester="dma", write=False,
            address=0, useful_bytes=4, ready_cycle=0, tag=None):
    return MemRequest(request_id, requester, write, address,
                      useful_bytes, ready_cycle, tag)


def drain(resource, limit=100):
    completed = []
    cycle = resource._last_cycle + 1
    while resource.pending and cycle < limit:
        completed.extend(resource.tick(cycle))
        cycle += 1
    if resource.pending:
        raise AssertionError("memory did not drain")
    return completed


class SharedMemoryTests(unittest.TestCase):
    def test_unaligned_odd_range_transfers_complete_aligned_beats(self):
        resource = memory(max_outstanding=2)
        original = request(address=3, useful_bytes=7, tag={"row": 3})
        resource.submit(original)
        self.assertEqual(drain(resource), [original])
        issues = [event for event in resource.trace if event["event"] == "beat_issue"]
        self.assertEqual([event["address"] for event in issues], [0, 4, 8])
        self.assertEqual([event["valid_bytes"] for event in issues], [1, 4, 2])
        self.assertEqual([event["valid_offset"] for event in issues], [3, 0, 0])
        self.assertEqual([event["cycle"] for event in issues], [0, 1, 2])
        self.assertTrue(all(event["tag"] == {"row": 3} for event in issues))
        stats = resource.stats()
        self.assertEqual(stats["useful_read_bytes"], 7)
        self.assertEqual(stats["physical_read_bytes"], 12)
        self.assertEqual(stats["completed_beats"], 3)
        self.assertEqual(stats["max_outstanding"], 2)
        self.assertEqual(stats["completed_requests"], 1)

    def test_outstanding_limit_is_per_beat_and_completions_free_slots_first(self):
        resource = memory(read_latency=3, max_outstanding=1)
        resource.submit(request(useful_bytes=12))
        drain(resource)
        issues = [event["cycle"] for event in resource.trace if event["event"] == "beat_issue"]
        self.assertEqual(issues, [0, 3, 6])
        self.assertEqual(resource.stats()["max_outstanding"], 1)
        self.assertEqual(resource.stats()["outstanding_stall_cycles"], 4)

    def test_round_robin_arbitration_is_deterministic(self):
        resource = memory(read_latency=1)
        resource.submit(request("a", "first", useful_bytes=12))
        resource.submit(request("b", "second", useful_bytes=12))
        drain(resource)
        issues = [event["requester"] for event in resource.trace if event["event"] == "beat_issue"]
        self.assertEqual(issues, ["first", "second"] * 3)
        self.assertEqual(resource.stats()["requester_first_issue_wait"],
                         {"first": 0, "second": 1})

    def test_requester_fifo_waits_for_acknowledgement(self):
        resource = memory(read_latency=1, write_latency=4)
        first = request("a", "dma", True, ready_cycle=3)
        second = request("b", "dma", False)
        resource.submit(first)
        resource.submit(second)
        self.assertEqual(drain(resource), [first, second])
        issues = [event for event in resource.trace if event["event"] == "beat_issue"]
        self.assertEqual([event["cycle"] for event in issues], [3, 7])
        self.assertEqual(resource.stats()["physical_write_bytes"], 4)
        self.assertEqual(resource.stats()["useful_write_bytes"], 4)

    def test_accept_interval_is_shared_between_requesters(self):
        resource = memory(read_latency=1, accept_interval=2)
        resource.submit(request("a", "first", useful_bytes=8))
        resource.submit(request("b", "second", useful_bytes=4))
        drain(resource)
        issues = [event["cycle"] for event in resource.trace if event["event"] == "beat_issue"]
        self.assertEqual(issues, [0, 2, 4])

    def test_split_channels_issue_read_and_write_in_same_cycle(self):
        resource = memory(read_write_shared=False, max_outstanding=2)
        resource.submit(request("r", "reader"))
        resource.submit(request("w", "writer", True))
        self.assertEqual(resource.tick(0), [])
        issues = [event for event in resource.trace if event["event"] == "beat_issue"]
        self.assertEqual([event["cycle"] for event in issues], [0, 0])
        self.assertEqual({event["channel"] for event in issues}, {"read", "write"})
        self.assertEqual(resource.tick(1), [])
        self.assertEqual([item.request_id for item in resource.tick(2)], ["r"])
        self.assertEqual([item.request_id for item in resource.tick(3)], ["w"])

    def test_split_channels_share_global_outstanding_limit_without_starvation(self):
        resource = memory(read_latency=1, write_latency=1,
                          read_write_shared=False, max_outstanding=1)
        resource.submit(request("r", "reader", useful_bytes=12))
        resource.submit(request("w", "writer", True, useful_bytes=12))
        drain(resource)
        issues = [event["write"] for event in resource.trace if event["event"] == "beat_issue"]
        self.assertEqual(issues, [False, True] * 3)
        self.assertEqual(resource.stats()["max_outstanding"], 1)

    def test_background_uses_accept_slots_without_claiming_requester_bytes(self):
        resource = memory(read_latency=1, background_period=2)
        resource.submit(request(useful_bytes=12))
        drain(resource)
        issues = [event["cycle"] for event in resource.trace if event["event"] == "beat_issue"]
        self.assertEqual(issues, [0, 2, 4])
        self.assertEqual(resource.stats()["background_slots"], 3)
        self.assertEqual(resource.stats()["background_bytes"], 12)
        self.assertEqual(resource.stats()["physical_read_bytes"], 12)
        self.assertEqual(resource.stats()["accepted_beats"], 3)

    def test_bounds_duplicate_ids_and_invalid_parameters(self):
        resource = memory(capacity_bytes=16)
        for bad in (request(address=-1), request(address=15, useful_bytes=2),
                    request(useful_bytes=0), request(ready_cycle=-1)):
            with self.assertRaises(ValueError):
                resource.submit(bad)
        resource.submit(request("last", address=15, useful_bytes=1))
        with self.assertRaises(ValueError):
            resource.submit(request("last"))
        drain(resource)
        self.assertEqual(resource.stats()["useful_read_bytes"], 1)
        for options in ({"beat_bytes": 0}, {"read_latency": 0},
                        {"write_latency": 0}, {"max_outstanding": 0},
                        {"accept_interval": 0}, {"background_period": -1}):
            with self.assertRaises(ValueError):
                memory(**options)

    def test_tick_cannot_repeat_or_go_backwards(self):
        resource = memory()
        resource.tick(3)
        for bad in (3, 2, -1):
            with self.assertRaises(ValueError):
                resource.tick(bad)

    def test_trace_can_be_disabled_and_stats_are_independent_snapshots(self):
        resource = memory(capture_trace=False)
        resource.submit(request())
        drain(resource)
        self.assertEqual(resource.trace, [])
        snapshot = resource.stats()
        snapshot["requester_wait"]["dma"] = -99
        self.assertGreaterEqual(resource.stats()["requester_wait"]["dma"], 0)

    def test_future_ready_request_does_not_block_other_requesters(self):
        resource = memory(read_latency=1)
        resource.submit(request("future", "future", ready_cycle=5))
        resource.submit(request("now", "now"))
        self.assertEqual([item.request_id for item in drain(resource)], ["now", "future"])


class BankedSRAMTests(unittest.TestCase):
    def test_single_port_conflicts_are_atomic_across_banks(self):
        resource = BankedSRAM("sp", banks=2, depth_words=8)
        self.assertTrue(resource.try_access(0, [(0, 0)], []))
        self.assertFalse(resource.try_access(0, [(0, 1), (1, 1)], []))
        # The failed transaction did not consume bank one's port.
        self.assertTrue(resource.try_access(0, [], [(1, 1)]))
        self.assertEqual(resource.stats()["read_accesses"], 1)
        self.assertEqual(resource.stats()["write_accesses"], 1)
        self.assertEqual(resource.stats()["conflicts"], 1)

    def test_read_broadcast_deduplicates_within_and_across_calls(self):
        resource = BankedSRAM("sp", 1, 8)
        self.assertTrue(resource.try_access(0, [(0, 1), (0, 1)], []))
        self.assertTrue(resource.try_access(0, [(0, 1)], []))
        self.assertEqual(resource.stats()["read_accesses"], 1)
        self.assertEqual(resource.stats()["read_requests"], 3)
        self.assertEqual(resource.stats()["read_broadcasts"], 2)
        self.assertFalse(resource.try_access(0, [(0, 2)], []))

    def test_one_read_one_write_per_bank_at_distinct_addresses(self):
        resource = BankedSRAM("rw", 1, 8, "1r1w", latency=2)
        self.assertTrue(resource.try_access(0, [(0, 0)], [(0, 1)]))
        self.assertFalse(resource.try_access(0, [(0, 2)], []))
        self.assertEqual(resource.completion_cycle(0), 2)
        self.assertEqual(resource.trace[0]["complete_cycle"], 2)

    def test_same_address_read_write_is_illegal_in_every_port_mode(self):
        for mode in ("single", "1r1w", "2rw"):
            resource = BankedSRAM(mode, 1, 8, mode)
            self.assertFalse(resource.try_access(0, [(0, 1)], [(0, 1)]))
            self.assertTrue(resource.try_access(0, [], [(0, 1)]))
            self.assertFalse(resource.try_access(0, [(0, 1)], []))

    def test_two_rw_allows_two_reads_two_writes_or_mixed(self):
        resource = BankedSRAM("two", 1, 8, "2rw")
        self.assertTrue(resource.try_access(0, [(0, 0), (0, 1)], []))
        self.assertFalse(resource.try_access(0, [], [(0, 2)]))
        self.assertTrue(resource.try_access(1, [], [(0, 0), (0, 1)]))
        self.assertTrue(resource.try_access(2, [(0, 0)], [(0, 1)]))
        self.assertEqual(resource.stats()["per_bank"][0]["reads"], 3)
        self.assertEqual(resource.stats()["per_bank"][0]["writes"], 3)

    def test_duplicate_writes_rejected_atomically(self):
        resource = BankedSRAM("two", 1, 8, "2rw")
        self.assertFalse(resource.try_access(0, [], [(0, 1), (0, 1)]))
        self.assertTrue(resource.try_access(0, [], [(0, 1)]))
        self.assertFalse(resource.try_access(0, [], [(0, 1)]))
        self.assertEqual(resource.stats()["write_accesses"], 1)

    def test_new_cycle_releases_ports_and_capacity_is_two_bytes_per_word(self):
        resource = BankedSRAM("sp", 2, 8)
        self.assertTrue(resource.try_access(0, [(0, 0)], []))
        self.assertTrue(resource.try_access(1, [(0, 1)], []))
        self.assertEqual(resource.capacity_bytes, 32)
        with self.assertRaises(ValueError):
            resource.try_access(0, [], [])

    def test_bad_addresses_and_configuration_rejected_without_grants(self):
        resource = BankedSRAM("sp", 2, 8)
        for bad in ([(-1, 0)], [(2, 0)], [(0, 8)], [(0, -1)], [(0,)]):
            with self.assertRaises(ValueError):
                resource.try_access(0, bad, [])
        self.assertEqual(resource.stats()["attempts"], 0)
        for options in ({"banks": 0}, {"depth_words": 0},
                        {"port_mode": "unknown"}, {"latency": 0}):
            params = dict(name="bad", banks=1, depth_words=8)
            params.update(options)
            with self.assertRaises(ValueError):
                BankedSRAM(**params)

    def test_trace_can_be_disabled_and_stats_copies_per_bank_counters(self):
        resource = BankedSRAM("sp", 1, 8, capture_trace=False)
        resource.try_access(0, [(0, 1)], [])
        self.assertEqual(resource.trace, [])
        snapshot = resource.stats()
        snapshot["per_bank"][0]["reads"] = 99
        self.assertEqual(resource.stats()["per_bank"][0]["reads"], 1)

    def test_sparse_accesses_preserve_other_banks_and_expire_on_new_cycle(self):
        resource = BankedSRAM("many", 64, 8)
        self.assertTrue(resource.try_access(0, [(0, 0)], []))
        self.assertTrue(resource.try_access(0, [], [(63, 2)]))
        self.assertTrue(resource.try_access(0, [(1, 3)], []))
        self.assertFalse(resource.try_access(0, [(0, 1)], []))
        self.assertFalse(resource.try_access(0, [], [(63, 3)]))
        self.assertTrue(resource.try_access(1, [], [(0, 1)]))
        self.assertTrue(resource.try_access(1, [(63, 2)], []))
        self.assertTrue(resource.try_access(1, [(1, 4)], []))
        banks = resource.stats()["per_bank"]
        self.assertEqual(banks[0], {"reads": 1, "writes": 1, "conflicts": 1})
        self.assertEqual(banks[63], {"reads": 1, "writes": 1, "conflicts": 1})
        self.assertEqual(banks[1], {"reads": 2, "writes": 0, "conflicts": 0})
        self.assertTrue(all(item == {"reads": 0, "writes": 0, "conflicts": 0}
                            for item in banks[2:63]))

    def test_failed_future_transaction_does_not_advance_or_clear_ports(self):
        resource = BankedSRAM("atomic", 64, 8)
        self.assertTrue(resource.try_access(0, [(0, 0)], []))
        before = resource.stats()
        before_trace = list(resource.trace)
        with self.assertRaises(ValueError):
            resource.try_access(1, [(0, 0), (63, 8)], [])
        self.assertEqual(resource.stats(), before)
        self.assertEqual(resource.trace, before_trace)
        self.assertFalse(resource.try_access(1, [(0, 0), (0, 1)], []))
        self.assertTrue(resource.try_access(0, [(0, 0)], []))
        self.assertFalse(resource.try_access(0, [(0, 1)], []))
        self.assertTrue(resource.try_access(0, [], [(63, 1)]))

    def test_empty_successful_new_cycle_expires_all_old_bank_ports(self):
        resource = BankedSRAM("empty", 64, 8)
        self.assertTrue(resource.try_access(0, [], [(63, 0)]))
        self.assertTrue(resource.try_access(1, [], []))
        self.assertTrue(resource.try_access(1, [], [(63, 0)]))
        self.assertFalse(resource.try_access(1, [(63, 1)], []))


if __name__ == "__main__":
    unittest.main()
