import csv, tempfile, unittest
from pathlib import Path
import torch
from src.benchmark_support import aggregate
from scripts.run_benchmark import prepared_world, combine

class BenchmarkTests(unittest.TestCase):
    def test_mean(self):
        m=[{'a':torch.tensor([float(i)])} for i in (1,2,9)]
        self.assertTrue(torch.allclose(aggregate(m,[1,2,1])['a'],torch.tensor([3.5])))
    def test_median(self):
        m=[{'a':torch.tensor([float(i)])} for i in (1,2,900)]
        self.assertTrue(torch.equal(aggregate(m,[1,1,1],median=True)['a'],torch.tensor([2.])))
    def test_nonfinite(self):
        m=[{'a':torch.tensor([float(i)])} for i in (1,2,float('nan'))]
        with self.assertRaises(ValueError): aggregate(m,[1,1,1])
    def test_world(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'test.wbt'
            original='WorldInfo {\n  randomSeed 42\n}\n'+'Robot { controller "frl_agent" }\n'*3+'Robot { controller "superviser" }\n'
            path.write_text(original); new=prepared_world(path,11)
            self.assertEqual(path.read_text(),original)
            self.assertIn('randomSeed 11',new.read_text()); self.assertEqual(new.read_text().count('"bench_agent"'),3)
            with self.assertRaises(FileExistsError): prepared_world(path,11)
    def test_combine(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d)
            for i in range(1,4):
                (folder/f'robot_{i}.csv').write_text(f'robot,interaction,reward\nrobot_{i},1,0.2\nrobot_{i},2,0.3\n')
            combine([folder],folder/'train.csv',True,2,2)
            with (folder/'train.csv').open() as f: self.assertEqual(len(list(csv.DictReader(f))),6)
            with self.assertRaises(ValueError): combine([folder],folder/'invalid.csv',True,3,2)

if __name__=='__main__': unittest.main()
