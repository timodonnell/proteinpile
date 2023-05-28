import prody
import proteopt.client

PDB_CACHE = {}
def load_pdb(filename):
    if filename not in PDB_CACHE:
        PDB_CACHE[filename] = prody.parsePDB(filename)
    return PDB_CACHE[filename]


def get_specification_variables(filename):
    with open(filename) as f:
        vars = {}
        code = compile(f.read(), filename, 'exec')
        exec(code, vars, vars)
        return vars


def get_runner(client, cls, local_conf={}, **kwargs):
    if client is not None:
        runner = client.remote_model(cls, **kwargs)
    else:
        runner = cls(**local_conf, **kwargs)
    return runner


def get_client(args):
    if args.endpoints_file:
        endpoints = []
        for f in args.endpoints_file:
            endpoints.extend(x.strip() + "/tool" for x in open(f).readlines() if x.strip())
        print("Using endpoints [%d]" % len(endpoints), endpoints)
        client = proteopt.client.Client(
            endpoints=endpoints,
            extra_parallelism_factor=args.client_extra_parallelism_factor)
    else:
        print("No endpoints, will run locally")
        client = None
    return client


PROBLEM_CACHE = {}
def get_problem(spec, params_dict):
    if params_dict not in PROBLEM_CACHE:
        PROBLEM_CACHE[params_dict] = spec.get_problem(params_dict).get_first_chain()
    return PROBLEM_CACHE[params_dict]