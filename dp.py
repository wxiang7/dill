import imp
import dill.dill.dill
import sys

sys.path.append('/Vault/workspace/github/airflow/dags')

m = imp.load_source('tset_mod', '/Vault/workspace/github/airflow/dags/testdruid.py')

print(dill.dill.dumps(m.dag, main=m))