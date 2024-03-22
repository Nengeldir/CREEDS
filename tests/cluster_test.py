import unittest
from creeds.cluster import Cluster

class TestCluster(unittest.TestCase):
    def test_cluster(self):
        cluster = Cluster()
        self.assertIsInstance(cluster, Cluster)

if __name__ == '__main__':
    unittest.main()
