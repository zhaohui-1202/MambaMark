from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if __name__ == '__main__':
    from evaluation.evaluate_distinguishability import main
    main(['--gallery', 'copyright', *sys.argv[1:]])
