import os

from girder import logger
from girder import plugin
from girder.api import access
from girder.api.describe import Description
from girder.api.describe import autoDescribeRoute
from girder.api.rest import Resource
from girder.exceptions import RestException


class Modsquad(Resource):
    def __init__(self):
        super(Modsquad, self).__init__()
        self.resourceName = 'modsquad'

        # Config
        self.route('GET', ('config',), self.getConfig)

        # Datasets.
        self.route('GET', ('dataset', 'data'), self.getDataset)
        self.route('GET', ('dataset', 'features'), self.getDatasetFeatures)
        self.route('GET', ('dataset', 'metadata'), self.getFeatureMetadata)
        self.route('GET', ('dataset', 'problems'), self.getProblems)

        # Pipelines.
        self.route('GET', ('pipeline', 'results'), self.getResults)
        self.route('POST', ('pipeline',), self.createPipeline)
        self.route('POST', ('pipeline', 'export'), self.executePipeline)
        self.route('POST', ('pipeline', 'results'), self.exportPipeline)

        # Stop process.
        self.route('POST', ('stop',), self.stopProcess)

    @access.public
    @autoDescribeRoute(
        Description('Foobar')
    )
    def getConfig(self):
        config_file = os.environ.get('JSON_CONFIG_PATH')

        logger.info('environment variable said: %s' % (config_file))
        #config_file = "/Users/clisle/proj/D3M/code/eval/config.json"
        logger.info('config service: looking for config file %s' % (config_file))

        if config_file is None:
            raise RestException('JSON_CONFIG_PATH is not set!', code=500)

        try:
            with open(config_file) as f:
                text = f.read()
        except IOError as e:
            raise RestException(str(e), code=500)

        try:
            config = json.loads(text)
        except ValueError as e:
            raise RestException('Could not parse JSON - %s' % (str(e)), code=500)

        # clean up the output filesystem to remove any files from old runs
        try:
            os.system('rm -rf /output/*')
            os.mkdir('/output/pipelines')
            os.mkdir('/output/executables')
            os.mkdir('/output/supporting_files')
            os.mkdir('/output/predictions')
        except:
            pass

        logger.info('received json configuration: %s' % (config))

        os.environ['PROBLEM_SCHEMA_PATH'] = config['problem_schema']
        os.environ['DATASET_SCHEMA_PATH'] = config['dataset_schema']
        os.environ['TRAINING_DATA_ROOT'] = config['training_data_root']
        os.environ['PROBLEM_ROOT'] = config['problem_root']
        os.environ['EXECUTABLES_ROOT'] = config['executables_root']
        # used by the ta2read service
        os.environ['TEMP_STORAGE_ROOT'] = config['temp_storage_root']

        return config

    @access.public
    @autoDescribeRoute(
        Description('Foobar')
    )
    def getDataset(self):
        problem_schema_path = os.environ.get('PROBLEM_ROOT')
        dataset_schema_path = os.environ.get('TRAINING_DATA_ROOT')
        datasupply = d3mds.D3MDS(dataset_schema_path,problem_schema_path)
        # fill nan with zeros, or should it be empty strings?
        data_as_df = datasupply.get_data_all().fillna('')
        list_of_dicts = copy.deepcopy(data_as_df.T.to_dict().values())
        #print 'train data excerpt: ',list_of_dicts
        #print 'end of data'
        return list_of_dicts

    @access.public
    @autoDescribeRoute(
        Description('Foobar')
    )
    def getDatasetFeatures(self):
        featurelist = []
        # get the data handy by reading it
        dataset = self.getDataset()

        # Iterate over first entry, assuming the data is uniform ( no missing fields)
        for feat in dataset[0].keys():
            featurelist.append(feat)
        return featurelist

    @access.public
    @autoDescribeRoute(
        Description('Foobar')
    )
    def getFeatureMetadata(self):
        dataset_schema_path = os.environ.get('TRAINING_DATA_ROOT')
        datasupply = d3mds.D3MDataset(dataset_schema_path)
        return datasupply.get_learning_data_columns()

    @access.public
    @autoDescribeRoute(
        Description('Foobar')
    )
    def getProblems(self):
        problem_schema_path = os.environ.get('PROBLEM_ROOT')
        problem_supply = d3mds.D3MProblem(problem_schema_path)
        targets = problem_supply.get_targets()
        metrics = problem_supply.get_performance_metrics()
        # taskType = problem_supply.get_taskType()
        # subType =  problem_supply.get_taskSubType()
        # targets = problem_supply.get_targets()
        problems = []
        problems.append({'problemId': problem_supply.get_problemID(),
                            #'description': problem_supply.get_problemDescription(),
                            # 'taskType': taskType,
                            # 'taskSubType': subType,
                             'metrics' : metrics,
                             'targets': problem_supply.get_targets()
                             })

        return problems

    @access.public
    @autoDescribeRoute(
        Description('Foobar')
    )
    def getResults(self):
        print 'copying pipelineURI:',resultURI
        if resultURI is None:
            raise RestException('no resultURI for executed pipeline', code=500)

        if resultURI[0:7] == 'file://':
            resultURI = resultURI[7:]

        # copy the results file under the webroot so it can be read by
        # javascript without having cross origin problems
        with open(resultURI,'r') as f:
          content = f.read()
          f.close()
          return content

    @access.public
    @autoDescribeRoute(
        Description('Foobar')
    )
    def createPipeline(self):
      stub = get_stub()

      problem_schema_path = os.environ.get('PROBLEM_ROOT')
      problem_supply = d3mds.D3MProblem(problem_schema_path)

      # get a pointer to the original dataset description doc
      dataset_schema_path = os.environ.get('TRAINING_DATA_ROOT')

      # if the user has elected to ignore some variables, then generate a modified spec
      # and load from the modified spec

      if inactive != None:
        print 'detected inactive variables:', inactive
        modified_dataset_schema_path = '/output/supporting_files'
        generate_modified_database_spec(dataset_schema_path,modified_dataset_schema_path, inactive)
        dataset_supply = d3mds.D3MDataset(modified_dataset_schema_path)
      else:
        dataset_supply = d3mds.D3MDataset(dataset_schema_path)

      # get the target features into the record format expected by the API
      targets =  problem_supply.get_targets()
      # features = []
      # for entry in targets:
        # tf = core_pb2.Feature(resource_id=entry['resID'],feature_name=entry['colName'])
        # features.append(tf)

      # we are having trouble parsing the problem specs into valid API specs, so just hardcode
      # to certain problem types for now.  We could fix this with a more general lookup table to return valid API codes
      # task = taskTypeLookup(task_type)
      # tasksubtype = subTaskLookup(task_subtype)

      # the metrics in the files are imprecise text versions of the enumerations, so just standardize.  A lookup table
      # would help here, too
      # metrics=[core_pb2.F1_MICRO, core_pb2.ROC_AUC, core_pb2.ROOT_MEAN_SQUARED_ERROR, core_pb2.F1, core_pb2.R_SQUARED]

      # context_in = cpb.SessionContext(session_id=context)

      # problem_pb = Parse(json.dumps(problem_supply.prDoc), problem_pb2.ProblemDescription(), ignore_unknown_fields=True)

      # currently HTTP timeout occurs after 2 minutes (probably from , so clamp this value to 2 minutes temporarily)
      #print 'clamping search time to 2 minutes to avoid timeouts'
      #time_limit = min(2,int(time_limit))
      

      problem = problem_pb2.Problem(
        id = problem_supply.get_problemID(),
        version = problem_supply.get_problemSchemaVersion(),
        name = 'modsquad_problem',
        description = 'modsquad problem',
        task_type = taskTypeLookup(problem_supply.get_taskType()),
        task_subtype = subTaskLookup(problem_supply.get_taskSubType()),
        performance_metrics = map(lambda x: problem_pb2.ProblemPerformanceMetric(metric=metricLookup(x['metric'], problem_supply.get_taskType())), problem_supply.get_performance_metrics()))

      value = value_pb2.Value(dataset_uri=data_uri)
      req = core_pb2.SearchSolutionsRequest(
              user_agent='modsquad',
              #version=core_pb2.protcol_version,
              version="2018.7.7",
              time_bound=int(time_limit),
              problem=problem_pb2.ProblemDescription(
                  problem=problem,
                  inputs=[problem_pb2.ProblemInput(
                      dataset_id=dataset_supply.get_datasetID(),
                      targets=map(make_target, problem_supply.get_targets()))]),
              inputs=[value])
      resp = stub.SearchSolutions(req)
      print 'set time bound to be: ',time_limit,' minutes'
      print 'using hard-coded version 2018.7.7 of the API. Should pull from the proto files instead'

      # return map(lambda x: json.loads(MessageToJson(x)), resp)
      search_id = json.loads(MessageToJson(resp))['searchId']

      # Get actual pipelines.
      req = core_pb2.GetSearchSolutionsResultsRequest(search_id=search_id)
      results = stub.GetSearchSolutionsResults(req)
      results = map(lambda x: json.loads(MessageToJson(x)), results)

      stub.StopSearchSolutions(core_pb2.StopSearchSolutionsRequest(search_id=search_id))

      return results

    @access.public
    @autoDescribeRoute(
        Description('Foobar')
    )
    def executePipeline(self):
        stub = get_stub()

        # add file descriptor if it is missing. some systems might be inconsistent, but file:// is the standard
        if data_uri[0:4] != 'file':
          data_uri = 'file://%s' % (data_uri)

        # context_in = cpb.SessionContext(session_id=context)

        input = value_pb2.Value(dataset_uri=data_uri)
        request_in = cpb.FitSolutionRequest(solution_id=pipeline,
                                            inputs=[input])
        resp = stub.FitSolution(request_in)

        resp = json.loads(MessageToJson(resp))
        pprint.pprint(resp)

        fittedPipes = stub.GetFitSolutionResults(core_pb2.GetFitSolutionResultsRequest(request_id=resp['requestId']))
        # print list(fittedPipes)
        # fittedPipes = map(lambda x: MessageToJson(x), fittedPipes)
        # for f in fittedPipes:
            # f['fittedSolutionId'] = json.loads(f['fittedSolutionId'])

        fittedPipes = list(fittedPipes)
        # map(pprint.pprint, fittedPipes)
        print 'fitted pipes:'
        map(lambda x: pprint.pprint(MessageToJson(x)), fittedPipes)

        pipes = []
        for f in fittedPipes:
            # f = json.loads(MessageToJson(f))
            #pprint.pprint(f)
            pipes.append(json.loads(MessageToJson(f)))

        fitted_solution_id = map(lambda x: x['fittedSolutionId'],filter(lambda x: x['progress']['state'] == 'COMPLETED', pipes))
        print 'fitted_solution_id', fitted_solution_id

        executedPipes = map(lambda x: stub.ProduceSolution(core_pb2.ProduceSolutionRequest(
            fitted_solution_id=x['fittedSolutionId'],
            inputs=[input])), filter(lambda x: x['progress']['state'] == 'COMPLETED', pipes))

        # executedPipes = map(lambda x: json.loads(MessageToJson(x)), executedPipes)
        #print 'executed pipes:'
        #pprint.pprint(executedPipes)

        results = map(lambda x: stub.GetProduceSolutionResults(core_pb2.GetProduceSolutionResultsRequest(request_id=x.request_id)), executedPipes)

        #print 'results is:'
        #pprint.pprint(results)
        exposed = []
        for r in results:
            for rr in r:
                #pprint.pprint(rr)
                #pprint.pprint(MessageToJson(rr))

                exposed.append(json.loads(MessageToJson(rr)))

        exposed = filter(lambda x: x['progress']['state'] == 'COMPLETED', exposed)
        #pprint.pprint(exposed)

        # the loop through the returned pipelines to copy their data
        # is not used anymore. Tngelo
        #map(lambda x: copyToWebRoot(x), exposed)
        return {'exposed': exposed, 'fitted_solution_id':fitted_solution_id}
        # magic saved here: return [{exposed: v[0], fitted_id: v[1]} for v in zip(exposed, fitted_solution_id)]

    @access.public
    @autoDescribeRoute(
        Description('Foobar')
    )
    def exportPipeline(self):
        global globalNextRankToUse
        stub = get_stub()

        # if there was a rank input to this call, use that rank, otherwise
        # increment a global ranked value each time this method is called.
        # This way, successfive exports will have increasing rank numbers (1,2,3,etc.)
        if rankInput:
          rankToOutput = int(rankInput)
          globalNextRankToUse = rankInput + 1
        else:
          rankToOutput = int(globalNextRankToUse)
          # increment the global counter so the next use will have a higher rank
          globalNextRankToUse += 1

        request_in = cpb.SolutionExportRequest(fitted_solution_id=pipeline,
                                            rank=int(rankToOutput))

        print 'requesting solution export:', request_in
        resp = stub.SolutionExport(request_in)

        return json.loads(MessageToJson(resp))

    @access.public
    @autoDescribeRoute(
        Description('Foobar')
    )
    def stopProcess(self):
        return {'foo': 'bar'}


class GirderPlugin(plugin.GirderPlugin):
    DISPLAY_NAME = 'Modsquad'

    def load(self, info):
        info['apiRoot'].modsquad = Modsquad()
