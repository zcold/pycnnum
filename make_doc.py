import pdoc.cli
import shlex
import sys

cmd = 'pdoc3 pycnnum --html -o doc -c "lunr_search={\'fuzziness\': 1, \'index_docstrings\': True}" -c "show_source_code=False" -c "sort_identifiers=False" -c "latex_math=True" --force'
sys.argv = shlex.split(cmd)
pdoc.cli.main()