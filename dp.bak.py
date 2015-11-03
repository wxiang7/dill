
def dump():

    import sys
    import imp
    from dill import dumps, extend
    from subprocess import Popen

    extend()

    sys.path.append('/Vault/workspace/github/airflow/dags')
    sys.path.append('/Vault/workspace/github/dill/venv')
    m = imp.load_source('tset_mod', '/Vault/workspace/github/dill/venv/test_ops.py')

    a = [1, 3]
    b = [2, 4]
    a[0] = b
    b[1] = a
    s = dumps(a)

    s = dumps(m.dag, ship_path=['/Vault/workspace/github/dill/venv/'])

    print 'encoded:'
    # print(repr(s))


    code = 'import dill\nimport sys\nx = dill.loads(%s)\ny = x.tasks[0].python_callable\nrepr(y(0))' % repr(s)
    print 'running:'
    print code
    sp = Popen(['/Vault/workspace/github/dill-origin/venv/bin/python', '-c', code])

    ret = sp.wait()
    print('returned:')
    print(ret)


def load():
    import dill
    x = dill.loads('\x80\x02cairflow.models\nDAG\nq\x00)\x81q\x01}q\x02(U\x11schedule_intervalq\x03cdatetime\ntimedelta\nq\x04K\x01K\x00K\x00\x87q\x05Rq\x06U\x05tasksq\x07]q\x08(cpython_operator\nPythonOperator\nq\t)\x81q\n}q\x0b(U\x0ctrigger_ruleq\x0cX\x0b\x00\x00\x00all_successq\rU\x0e_upstream_listq\x0e]q\x0fU\x05ownerq\x10U\x07airflowq\x11U\x11on_retry_callbackq\x12NU\x13wait_for_downstreamq\x13\x89U\top_kwargsq\x14}q\x15U\x05adhocq\x16\x89U\x06paramsq\x17}q\x18U\x10_downstream_listq\x19]q\x1ah\t)\x81q\x1b}q\x1c(h\x0ch\rh\x0e]q\x1dh\nah\x10h\x11h\x12Nh\x13\x89h\x14}q\x1eU\x0brandom_baseq\x1fK\x00sh\x16\x89h\x17}q h\x19]q!U\x13on_failure_callbackq"NU\x0fpriority_weightq#K\x01U\nstart_dateq$cdatetime\ndatetime\nq%U\n\x07\xdf\n\x1b\x00\x00\x00\x00\x00\x00q&\x85q\'Rq(U\x06dag_idq)U\x17example_python_operatorq*U\x03dagq+h\x01U\x0eemail_on_retryq,\x88U\x08end_dateq-NU\x0etemplates_dictq.NU\x07op_argsq/]q0U\x10email_on_failureq1\x88U\x13on_success_callbackq2NU\x0bretry_delayq3h\x04K\x00M,\x01K\x00\x87q4Rq5U\x04poolq6NU\x07retriesq7K\x00U\x11execution_timeoutq8NU\x0fprovide_contextq9\x89U\x0fpython_callableq:cdill.dill\n_create_function\nq;(cdill.dill\n_unmarshal\nq<U\xf8c\x01\x00\x00\x00\x01\x00\x00\x00\x02\x00\x00\x00C\x00\x01\x00s\x11\x00\x00\x00t\x00\x00j\x01\x00|\x00\x00\x83\x01\x00\x01d\x01\x00S(\x02\x00\x00\x00s9\x00\x00\x00This is a function that will run within the DAG executionN(\x02\x00\x00\x00t\x04\x00\x00\x00timet\x05\x00\x00\x00sleep(\x01\x00\x00\x00t\x0b\x00\x00\x00random_base(\x00\x00\x00\x00(\x00\x00\x00\x00s-\x00\x00\x00/Vault/workspace/github/dill/venv/test_ops.pyt\x14\x00\x00\x00my_sleeping_function\x15\x00\x00\x00s\x02\x00\x00\x00\x00\x02q=\x85q>Rq?}q@(U\x04taskqAh\x1bU\x03DAGqBh\x00U\x14my_sleeping_functionqCctset_mod\nmy_sleeping_function\nU\ttimedeltaqDh\x04U\x0eseven_days_agoqEh(U\x0c__builtins__qFc__builtin__\n__dict__\nU\x04argsqG}qH(h\x10h\x11h$h(uU\x08__file__qIU./Vault/workspace/github/dill/venv/test_ops.pycqJU\x06pprintqKcpprint\npprint\nqLU\x0b__package__qMNU\x0ePythonOperatorqNh\tU\x01iqOK\x00U\x05rangeqPcfuture.types.newrange\nnewrange\nqQh+h\x01U\rprint_contextqRh;(h<U\xf0c\x01\x00\x00\x00\x02\x00\x00\x00\x02\x00\x00\x00K\x00\x01\x00s\x0e\x00\x00\x00t\x00\x00d\x01\x00\x83\x01\x00\x01d\x02\x00S(\x03\x00\x00\x00Ni\x01\x00\x00\x00s,\x00\x00\x00Whatever you return gets printed in the logs(\x01\x00\x00\x00t\x14\x00\x00\x00my_sleeping_function(\x02\x00\x00\x00t\x02\x00\x00\x00dst\x06\x00\x00\x00kwargs(\x00\x00\x00\x00(\x00\x00\x00\x00s-\x00\x00\x00/Vault/workspace/github/dill/venv/test_ops.pyt\r\x00\x00\x00print_context\x1a\x00\x00\x00s\x04\x00\x00\x00\x00\x03\n\x01qS\x85qTRqUh@hRNN}qVU\x08tset_modqWtqXRqYU\x04timeqZcdill.dill\n_import_module\nq[U\x04timeq\\\x85q]Rq^U\x08__name__q_hWU\x08run_thisq`h\nU\x08datetimeqah%U\x07__doc__qbNU\x0eprint_functionqc(c__future__\n_Feature\nqdoqe}qf(U\tmandatoryqg(K\x03K\x00K\x00U\x05alphaqhK\x00tqiU\x08optionalqj(K\x02K\x06K\x00hhK\x02tqkU\rcompiler_flagqlJ\x00\x00\x01\x00ubuhCNN}qmhWtqnRqoU\x07task_idqpU\x0bsleep_for_0qqU\x06_compsqrc__builtin__\nset\nqs]qt(X\x11\x00\x00\x00schedule_intervalquX\x11\x00\x00\x00execution_timeoutqvX\x13\x00\x00\x00wait_for_downstreamqwX\x07\x00\x00\x00task_idqxX\x13\x00\x00\x00on_failure_callbackqyX\x05\x00\x00\x00ownerqzX\x05\x00\x00\x00adhocq{X\n\x00\x00\x00start_dateq|X\x05\x00\x00\x00emailq}X\x0b\x00\x00\x00retry_delayq~X\x13\x00\x00\x00on_success_callbackq\x7fX\x0f\x00\x00\x00depends_on_pastq\x80X\x11\x00\x00\x00on_retry_callbackq\x81X\x0f\x00\x00\x00priority_weightq\x82X\x03\x00\x00\x00slaq\x83X\x06\x00\x00\x00dag_idq\x84X\x0e\x00\x00\x00email_on_retryq\x85e\x85q\x86Rq\x87U\x05emailq\x88NU\x05queueq\x89cfuture.types.newstr\nnewstr\nq\x8aX\x07\x00\x00\x00defaultq\x8b\x85q\x8c\x81q\x8d}q\x8ebU\x12_schedule_intervalq\x8fNU\x0fdepends_on_pastq\x90\x89U\x03slaq\x91Nubah"Nh#K\x01h$h(h)h*h+h\x01h,\x88h-Nh.Nh/]q\x92h1\x88h2Nh3h5h6Nh7K\x00h8Nh9\x88h:hYhpU\x11print_the_contextq\x93hrhs]q\x94(huhvhwhxhyhzh{h|h}h~h\x7fh\x80h\x81h\x82h\x83h\x84h\x85e\x85q\x95Rq\x96h\x88Nh\x89h\x8dh\x8fNh\x90\x89h\x91Nubh\x1beU\x0blast_loadedq\x97h%U\n\x07\xdf\x0b\x03\x01$\x01\x055 q\x98\x85q\x99Rq\x9ah-h%U\n\x07\xdf\x0b\x03\x01$\x01\x055\x1cq\x9b\x85q\x9cRq\x9dhrhs]q\x9e(X\x11\x00\x00\x00schedule_intervalq\x9fX\x05\x00\x00\x00tasksq\xa0X\n\x00\x00\x00parent_dagq\xa1X\x0b\x00\x00\x00last_loadedq\xa2X\r\x00\x00\x00full_filepathq\xa3X\x13\x00\x00\x00template_searchpathq\xa4X\n\x00\x00\x00start_dateq\xa5X\x06\x00\x00\x00dag_idq\xa6e\x85q\xa7Rq\xa8U\x0bsafe_dag_idq\xa9X\x17\x00\x00\x00example_python_operatorq\xaaU\x13user_defined_macrosq\xabNU\nparent_dagq\xacNh\x17NU\ntask_countq\xadK\x02U\rfull_filepathq\xaeX\x00\x00\x00\x00q\xafU\x0cdefault_argsq\xb0hHU\x13template_searchpathq\xb1Nh$Nh)h*ub.')
    print repr(x.tasks[0].python_callable(0))


dump()
