# -*- coding: utf-8 -*-
"""Assemble les 5 parties en PROJET_RAG_COMPLET.ipynb."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from nb_helpers import write_notebook
from nb_part1 import part1
from nb_part2 import part2
from nb_part3 import part3
from nb_part4 import part4
from nb_part5 import part5

cells = part1 + part2 + part3 + part4 + part5
write_notebook(cells, "PROJET_RAG_COMPLET.ipynb")
