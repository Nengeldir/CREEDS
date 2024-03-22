import unittest
from cluster_maker import ClusterMaker

class TestCluster(unittest.TestCase):
    def test_cluster(self):
        cluster = ClusterMaker("../input/setCD")
        self.assertIsInstance(cluster, ClusterMaker)

if __name__ == '__main__':
    unittest.main()
#test 123
