"""Check that the renamed source tree and Hydra targets remain connected."""

import ast
from importlib.util import resolve_name
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'OMP'


class ReleaseLayoutTests(unittest.TestCase):
    def test_python_sources_and_local_imports_resolve(self):
        for path in SOURCE.rglob('*.py'):
            with self.subTest(path=path.relative_to(ROOT)):
                tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom):
                        if node.level:
                            package = '.'.join(path.relative_to(SOURCE).parts[:-1])
                            modules = [resolve_name('.' * node.level + (node.module or ''), package)]
                        else:
                            modules = [node.module] if node.module else []
                    elif isinstance(node, ast.Import):
                        modules = [item.name for item in node.names]
                    else:
                        continue
                    for module in modules:
                        if module.split('.')[0] not in {'omp', 'baselines'}:
                            continue
                        target = SOURCE.joinpath(*module.split('.'))
                        self.assertTrue(target.is_dir() or target.with_suffix('.py').is_file(),
                                        f'{path}:{node.lineno}: {module}')

    def test_hydra_targets_exist_after_rename(self):
        for path in SOURCE.rglob('*.yaml'):
            for target in re.findall(r'^\s*_target_:\s*(\S+)', path.read_text(), re.M):
                if target.split('.')[0] not in {'omp', 'baselines'}:
                    continue
                with self.subTest(config=path.relative_to(ROOT), target=target):
                    module, symbol = target.rsplit('.', 1)
                    source = SOURCE.joinpath(*module.split('.')).with_suffix('.py')
                    self.assertTrue(source.is_file(), str(source))
                    tree = ast.parse(source.read_text(encoding='utf-8'))
                    symbols = {node.name for node in tree.body
                               if isinstance(node, (ast.ClassDef, ast.FunctionDef))}
                    self.assertIn(symbol, symbols)

    def test_release_contains_only_requested_policy_variants(self):
        expected = {
            'dp3': ('dp3', 'DP3'),
            'flowpolicy': ('flowpolicy', 'FlowPolicy'),
            'mp1': ('mp1', 'MP1'),
            'omp_jvp': ('omp_jvp_policy', 'OMPJVPPolicy'),
            'omp_dde': ('omp_dde_policy', 'OMPDDEPolicy'),
        }
        self.assertEqual(
            {path.stem for path in (SOURCE / 'omp/policy').glob('*.py')},
            {'base_policy'} | {module for module, _ in expected.values()},
        )
        self.assertEqual(
            {path.stem for path in (SOURCE / 'omp/config').glob('*.yaml')},
            set(expected),
        )
        for variant, (module, policy_class) in expected.items():
            config = (SOURCE / f'omp/config/{variant}.yaml').read_text(encoding='utf-8')
            self.assertIn(f'_target_: omp.policy.{module}.{policy_class}', config)

    def test_active_sources_have_no_private_machine_paths(self):
        for path in list(SOURCE.rglob('*.py')) + list(SOURCE.rglob('*.yaml')) + list((ROOT / 'scripts').glob('*.sh')):
            with self.subTest(path=path.relative_to(ROOT)):
                self.assertNotRegex(path.read_text(encoding='utf-8'),
                                    r'/bd_[^\s\"\']*|/home/[^\s\"\']+|/data/code/')


if __name__ == '__main__':
    unittest.main()
