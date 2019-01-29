import os

from girder import logger
from girder import plugin
from girder.api import access
from girder.api.describe import Description
from girder.api.describe import autoDescribeRoute
from girder.api.rest import Resource
from girder.exceptions import RestException
import json
from . import d3mds
import copy

# for the external girder dataset access
import girder_client
import requests
import json
from pandas.compat import StringIO
import pandas as pd
import requests

# for the TA2/TA3 api
import grpc
from google.protobuf.json_format import MessageToJson
from google.protobuf.json_format import Parse
from . import pipeline_pb2
from . import problem_pb2
from . import value_pb2
from . import primitive_pb2
from . import core_pb2
from . import core_pb2_grpc

# for the data import
import StringIO
import pandas as pd
import numpy as  np

# for generating specs
from . import generateSpecs


# girder address to read datasets from
girder_api_prefix = 'http://localhost:8080/api/v1'

dynamic_problem_root = '/output/modsquad_files'

# datamart API to use for suggesting new datasets
datamart_api_prefix = ''

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

        # upload from custom interfaces
        self.route('POST', ('dataset', 'upload'), self.uploadGeoappDataset)

        # added Jan 2019
        self.route('GET', ('dataset', 'external_list'), self.getExternalDatasetList)
        self.route('GET', ('dataset', 'external_download'), self.getExternalFileContents)
        self.route('GET', ('dataset', 'merge'), self.pairwiseDatasetMerge)
        self.route('GET', ('dataset', 'merge_and_store'), self.pairwiseDatasetMergeWithStorage)

        # Pipelines.
        self.route('GET', ('pipeline', 'results'), self.getResults)
        self.route('POST', ('pipeline',), self.createPipeline)
        self.route('POST', ('pipeline', 'execute'), self.executePipeline)
        self.route('POST', ('pipeline', 'export'), self.exportPipeline)

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


    def returnNumberIfConvertible(self,value):
      try:
        num = float(value)
        return num
      except ValueError:
        return value

    # look through an array of dictionaries and convert any numeric strings
    # (e.g. "245") to their numeric equivalient 
    def convertNumberStringsToNumeric(self,data):
      for row in data:
        for key in row.keys():
          row[key] = self.returnNumberIfConvertible(row[key])
      return data


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
        #list_of_dicts = data_as_df.T.to_dict()
        list_of_dicts = [data_as_df.iloc[line,:].T.to_dict() for line in range(len(data_as_df))]

        # need a persistent allocated copy for the front end to use?
        copy_of_data = copy.deepcopy(list_of_dicts)
        return self.convertNumberStringsToNumeric(copy_of_data)

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


    def get_stub(self):
        server_channel_address = os.environ.get('TA2_SERVER_CONN')
        # complain in the return if we didn't get an address to connect to
        if server_channel_address is None:
            logger.error('no TA2 server details to use for connection')
            return {'error': 'TA2_SERVER_CONN environment variable is not set!'}
        channel = grpc.insecure_channel(server_channel_address)
        stub = core_pb2_grpc.CoreStub(channel)
        return stub


    def metricLookup(self,metricString, task_type):
      # classification metrics
      if (metricString == 'accuracy'):
        print('accuracy metric')
        return problem_pb2.ACCURACY
      if (metricString == 'f1'):
        print('f1 metric')
        return problem_pb2.F1
      if (metricString == 'f1Macro'):
        print('f1-macro metric')
        return problem_pb2.F1_MACRO
      if (metricString == 'f1Micro'):
        print('f1-micro metric')
        return problem_pb2.F1_MICRO
      if (metricString == 'ROC_AUC'):
        print('roc-auc metric')
        return problem_pb2.ROC_AUC
      if (metricString == 'rocAuc'):
        print('rocAuc metric')
        return problem_pb2.ROC_AUC
      if (metricString == 'rocAucMicro'):
        print('roc-auc-micro metric')
        return problem_pb2.ROC_AUC_MICRO
      if (metricString == 'rocAucMacro'):
        print('roc-auc-macro metric')
        return problem_pb2.ROC_AUC_MACRO
      # clustering
      if (metricString == 'normalizedMutualInformation'):
        print('normalized mutual information metric')
        return problem_pb2.NORMALIZED_MUTUAL_INFORMATION
      if (metricString == 'jaccardSimilarityScore'):
        print('jaccard similarity metric')
        return problem_pb2.JACCARD_SIMILARITY_SCORE
      # regression
      if (metricString == 'meanSquaredError'):
        print('MSE metric')
        return problem_pb2.MEAN_SQUARED_ERROR
      if (metricString == 'rootMeanSquaredError'):
        print('RMSE metric')
        return problem_pb2.ROOT_MEAN_SQUARED_ERROR
      if (metricString == 'rootMeanSquaredErrorAvg'):
        print('RMSE Average metric')
        return problem_pb2.ROOT_MEAN_SQUARED_ERROR_AVG
      if (metricString == 'rSquared'):
        print('rSquared metric')
        return problem_pb2.R_SQUARED
      if (metricString == 'meanAbsoluteError'):
        print('meanAbsoluteError metric')
        return problem_pb2.MEAN_ABSOLUTE_ERROR
      # we don't recognize the metric, assign a value to the unknown metric according to the task type.
      else:
        print('undefined metric received, so assigning a metric according to the task type')
        if task_type==problem_pb2.CLASSIFICATION:
          print('classification: assigning f1Macro')
          return problem_pb2.F1_MACRO
        elif task_type==problem_pb2.CLUSTERING:
          print('clustering: assigning normalized mutual information')
          return problem_pb2.NORMALIZED_MUTUAL_INFORMATION
        else:
          print('regression: assigning RMSE')
          return problem_pb2.ROOT_MEAN_SQUARED_ERROR

    def make_target(self,spec):
        return problem_pb2.ProblemTarget(
                target_index = spec['targetIndex'],
                resource_id = spec['resID'],
                column_index = spec['colIndex'],
                column_name = spec['colName'])

    def taskTypeLookup(self,task):
      if (task=='classification'):
        print('detected classification task')
        return problem_pb2.CLASSIFICATION
      elif (task == 'clustering'):
        print('detected clustering task')
        return problem_pb2.CLUSTERING
      elif (task == 'objectDetection'):
        print('detected object detection task')
        return problem_pb2.OBJECT_DETECTION
      else:
        print('assuming regression')
        return problem_pb2.REGRESSION

    def subTaskLookup(self,sub):
      if (sub == 'multiClass'):
        print('multiClass subtype')
        return problem_pb2.MULTICLASS
      if (sub == 'multivariate'):
        return problem_pb2.MULTIVARIATE
      if (sub == 'univariate'):
        return problem_pb2.UNIVARIATE
      else:
        print('assuming NONE subtask')
        return problem_pb2.NONE


    # process the spec file and generate a new one with any inactive variables not included
    def generate_modified_database_spec(self,original,modified,inactive):
      # read the schema in dsHome
      _dsDoc = os.path.join(original, 'datasetDoc.json')
      assert os.path.exists(_dsDoc)
      with open(_dsDoc, 'r') as f:
        dsDoc = json.load(f)
        outDoc = {}
        outDoc['about'] = dsDoc['about']
        # loop through the resources and add them to the output spec if the feature is not inactive
        outDoc['dataResources'] = []
        for resource in dsDoc['dataResources']:
          # We moved only the dataset spec, update the paths to have an absolute path to the
          # original content
          resource['resPath'] = os.path.join(original, resource['resPath'])
          # pass things besides tables through automatically. tables have a list of features
          if resource['resType'] != 'table':
            outDoc['dataResources'].append(resource)
          else:
            # if it is a table, copy the header information, but clear out the column names and only
            # add columns that are not listed in the inactive list.  Inactive entries won't be added. 
            resourceOut = copy.deepcopy(resource)
            resourceOut['columns'] = []
            for column in resource['columns']:
              if column['colName'] not in inactive:
                # pass this feature record to the output columns 
                resourceOut['columns'].append(column)
            outDoc['dataResources'].append(resourceOut)
        # now the updated dataset spec will be written out to the write-enabled new location
        outFileName = os.path.join(modified, 'datasetDoc.json')
        assert os.path.exists(_dsDoc)
        with open(outFileName,'w') as outfile:
          json.dump(outDoc, outfile)

    # TA2 has written the results of a pipeline execution out to a file in a location
    # not readable by the modsquad front-end.  Read it here and return the file contents
    # as the result of the ajax call.

    @access.public
    @autoDescribeRoute(
        Description('Read computation fit results and return them through the ajax call for use by a client or a user interface')
    )
    def getResults(self,params):
        self.requireParams('resultURI', params)
        resultURI = params['resultURI']
        print('copying pipelineURI:',resultURI)
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
        Description('Send a problem description to an external modeling engine. Ask it to prepare to fit models for this problem')
    )
    def createPipeline(self,params):
      self.requireParams('data_uri', params)
      self.requireParams('inactive', params)
      self.requireParams('target',   params)
      self.requireParams('dynamic_mode', params)
      data_uri = params['data_uri']
      inactive = params['inactive']
      target = params['target']
      dynamic_mode = params['dynamic_mode']

      if dyamic_mode == True:
        # we are being called after the user built a custom dataset, don't use 
        # the environment variables.  Pull from the problemSpec instead
        problem_spec = generateSpecs.readProblemSpecFile(dynamic_problem_root,target)
        database_spec = generateSpecs.readDatasetDocFile(dynamic_problem_root)
        data_uri = dynamic_problem_root+'/datasetDoc.json'
    
        if 'time_limit' not in params:
          time_limit=1
        else:
          time_limit = params['time_limit']

        stub = self.get_stub()

        # if the user has elected to ignore some variables, then generate a modified spec
        # and load from the modified spec

        if inactive != None:
          print('detected inactive variables:', inactive)
          modified_dataset_schema_path = '/output/modsquad_modified_files'
          self.generate_modified_database_spec(dynamic_problem_root,modified_dataset_schema_path, inactive)
          dataset_spec = generateSpecs.readDatasetDocFile(modified_dataset_schema_path)

      
        # get the target features into the record format expected by the API
        targets =  problem_spec['inputs']['data'][0]['targets']  

        problem = problem_pb2.Problem(
          id = problem_spec['about']['problemID'],
          version = problem_spec['about']['problemSchemaVersion'],
          name = 'modsquad_problem',
          description = 'modsquad problem',
          task_type = self.taskTypeLookup(problem_spec['about']['taskType']),
          task_subtype = problem_pb2.NONE,
          performance_metrics = map(lambda x: problem_pb2.ProblemPerformanceMetric(metric=self.metricLookup(x['metric'], problem_spec['about']['taskType'])), problem_spec['inputs']['performanceMetrics']))

        value = value_pb2.Value(dataset_uri=data_uri)
        req = core_pb2.SearchSolutionsRequest(
                user_agent='modsquad',
                #version=core_pb2.protcol_version,
                version="2018.7.7",
                time_bound=int(time_limit),
                problem=problem_pb2.ProblemDescription(
                    problem=problem,
                    inputs=[problem_pb2.ProblemInput(
                        dataset_id=dataset_spec['about']['datasetID'],
                        targets=map(self.make_target, targets))]),
                inputs=[value])

      # use the standard way, read from files
      else:

        data_uri = params['data_uri']
        inactive = params['inactive']
        if 'time_limit' not in params:
          time_limit=1
        else:
          time_limit = params['time_limit']

        stub = self.get_stub()

        problem_schema_path = os.environ.get('PROBLEM_ROOT')
        problem_supply = d3mds.D3MProblem(problem_schema_path)
        # get a pointer to the original dataset description doc
        dataset_schema_path = os.environ.get('TRAINING_DATA_ROOT')

        # if the user has elected to ignore some variables, then generate a modified spec
        # and load from the modified spec

        if inactive != None:
          print('detected inactive variables:', inactive)
          modified_dataset_schema_path = '/output/modsquad_modified_files'
          self.generate_modified_database_spec(dataset_schema_path,modified_dataset_schema_path, inactive)
          dataset_supply = d3mds.D3MDataset(modified_dataset_schema_path)
        else:
          dataset_supply = d3mds.D3MDataset(dataset_schema_path)

        # get the target features into the record format expected by the API
        targets =  problem_supply.get_targets()  

        problem = problem_pb2.Problem(
          id = problem_supply.get_problemID(),
          version = problem_supply.get_problemSchemaVersion(),
          name = 'modsquad_problem',
          description = 'modsquad problem',
          task_type = self.taskTypeLookup(problem_supply.get_taskType()),
          task_subtype = self.subTaskLookup(problem_supply.get_taskSubType()),
          performance_metrics = map(lambda x: problem_pb2.ProblemPerformanceMetric(metric=self.metricLookup(x['metric'], problem_supply.get_taskType())), problem_supply.get_performance_metrics()))

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
                        targets=map(self.make_target, problem_supply.get_targets()))]),
                inputs=[value])


      #logger.info('about to make searchSolutions request')
      logger.info("sending search solutions request:",MessageToJson(req))
      resp = stub.SearchSolutions(req)
      #logger.info('after searchSolutionsRequest')
      print('set time bound to be: ',time_limit,' minutes')
      print('using hard-coded version 2018.7.7 of the API. Should pull from the proto files instead')

      # return map(lambda x: json.loads(MessageToJson(x)), resp)
      search_id = json.loads(MessageToJson(resp))['searchId']
      logger.info('after received search solutions answer')

      # Get actual pipelines.
      req = core_pb2.GetSearchSolutionsResultsRequest(search_id=search_id)
      #logger.info('sent search solutions results request')
      results = stub.GetSearchSolutionsResults(req)
      #logger.info('after received get solutions results')
      results = map(lambda x: json.loads(MessageToJson(x)), results)

      stub.StopSearchSolutions(core_pb2.StopSearchSolutionsRequest(search_id=search_id))
      return results


  

    @access.public
    @autoDescribeRoute(
        Description('This forces the modeling engine to fit a solution to provided data and returns the predictions.')
    )
    def executePipeline(self,params):
        self.requireParams('pipeline', params)
        self.requireParams('data_uri', params)
        pipeline = params['pipeline']
        data_uri = params['data_uri']
        
        stub = self.get_stub()

        # add file descriptor if it is missing. some systems might be inconsistent, but file:// is the standard
        if data_uri[0:4] != 'file':
          data_uri = 'file://%s' % (data_uri)

        # context_in = cpb.SessionContext(session_id=context)

        input = value_pb2.Value(dataset_uri=data_uri)
        request_in = core_pb2.FitSolutionRequest(solution_id=pipeline,
                                            inputs=[input])
        resp = stub.FitSolution(request_in)

        resp = json.loads(MessageToJson(resp))
        #pprint.pprint(resp)

        fittedPipes = stub.GetFitSolutionResults(core_pb2.GetFitSolutionResultsRequest(request_id=resp['requestId']))
        # print list(fittedPipes)
        # fittedPipes = map(lambda x: MessageToJson(x), fittedPipes)
        # for f in fittedPipes:
            # f['fittedSolutionId'] = json.loads(f['fittedSolutionId'])

        fittedPipes = list(fittedPipes)
        # map(pprint.pprint, fittedPipes)
        #print('fitted pipes:')
        #map(lambda x: pprint.pprint(MessageToJson(x)), fittedPipes)

        pipes = []
        for f in fittedPipes:
            # f = json.loads(MessageToJson(f))
            #pprint.pprint(f)
            pipes.append(json.loads(MessageToJson(f)))

        fitted_solution_id = map(lambda x: x['fittedSolutionId'],filter(lambda x: x['progress']['state'] == 'COMPLETED', pipes))
        print('fitted_solution_id', fitted_solution_id)

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
        Description('Signalthe modeling engine to export solutions as readable/reviewable files')
    )
    def exportPipeline(self):
        global globalNextRankToUse
        stub = self.get_stub()

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

        request_in = core_pb2.SolutionExportRequest(fitted_solution_id=pipeline,
                                            rank=int(rankToOutput))

        print('requesting solution export:', request_in)
        resp = stub.SolutionExport(request_in)
        return json.loads(MessageToJson(resp))


    @access.public
    @autoDescribeRoute(
        Description('Foobar')
    )
    def stopProcess(self):
        return {'foo': 'bar'}

    # this routine accesses an external girder interface and pulls out a list of all the 
    # files that are attached to items stored by this instance.  It returns the name and
    # girder Id for each file.  This information can be used to extract the file from girder.
    # girder authentication as 'admin' is attempted.  If it fails, only public files will 
    # be included in the returned list.

    @access.public
    @autoDescribeRoute(
        Description('Retreive a list of potential datasets stored in a local girder instance.')
    )
    def getExternalDatasetList(self):
      fileIdList = []
      gc = girder_client.GirderClient(apiUrl=girder_api_prefix)
      login = gc.authenticate('modsquad', 'd3md3m')
      #login = gc.authenticate('admin', 'd3md3m')
      collectionlist = gc.sendRestRequest('GET','collection')
      for coll in collectionlist:
          # look in each collection for all folders
          #print('looking in collection:',coll['name'])
          folderlist = gc.sendRestRequest('GET','folder',{'parentType':'collection','parentId':coll['_id']})
          for folder in folderlist:
              #print('looking in folder')
              itemlist = gc.sendRestRequest('GET','item',{'folderId':folder['_id']})
              for item in itemlist:
                  filelist = gc.sendRestRequest('GET','item/'+item['_id']+'/files')
                  for file in filelist:
                      #print('found filename:',file['name'], ' with ID:',file['_id'])
                      fileIdList.append(file)
      # Also report datasets stored in the dataManagerLite
      #for obj in listLiteDatasets():
      #  fileIdList.append(obj)

      # return a single object that contains a list of all discovered datasets
      retobj = {}
      retobj['data'] = fileIdList
      return retobj

    # this endpoint retrieves the contents of a file from girder, given a file Id (which 
    # is returned by the getExternalDatasetList method).  This assumes the file can be 
    # transported through an HTTP text response.   This doesn't currently authenticate to
    # girder, so publicly available files are assumed. 

    @access.public
    @autoDescribeRoute(
        Description('Retreive and return the contents of a dataset stored in a local girder instance.')
         .param('fileId', 'the girder Id of the file to retrieve', required=True,paramType='query')
    )
    def getExternalFileContents_old(self,params):
      self.requireParams('fileId', params)
      fileId = params['fileId']
      requesturl = girder_api_prefix+"/file/"+fileId+"/download"
      retobj = {}
      retobj['data'] = ''
      try:
        resp = requests.get(requesturl)
        #print(resp.text)
        retobj['data'] = resp.text
        return retobj
      except:
        # return an empty file if there was a problem with the read
        return retobj


    @access.public
    @autoDescribeRoute(
        Description('Retreive and return the contents of a dataset stored in a local girder instance.')
         .param('fileId', 'the girder Id of the file to retrieve', required=True,paramType='query')
    )
    def getExternalFileContents(self,params):
      self.requireParams('fileId', params)
      fileId = params['fileId']
      requesturl = girder_api_prefix+"/file/"+fileId+"/download"
      retobj = {}
      retobj['data'] = ''
      #try:
      resp = requests.get(requesturl)
      print(resp.text)

      # convert the file contents to a dataframe, so we can return it
      data_file = StringIO.StringIO(resp.text)
      data_df = pd.read_csv(data_file, sep=',')
      dataset_contents = data_df.to_dict('records')
      dataset_row_count = data_df.shape[0]
      dataset_column_count = data_df.shape[1]
   
      # generate metadata
      (dataset_typelist,labels) = generateSpecs.generate_datatypes(data_df)
 
      # generate and write out new specs for dataset and problem
      full_problem_spec = generateSpecs.generate_dynamic_problem_spec(data_df,dataset_typelist)
      database_spec = generateSpecs.generate_database_spec(full_problem_spec,data_df)
      generateSpecs.writeDatabaseDocFile(dynamic_problem_root, database_spec)
      generateSpecs.writeDatasetContents(dynamic_problem_root, database_spec, data_df)
      generateSpecs.writeProblemSpecFile(dynamic_problem_root, full_problem_spec)

      #retobj['data'] = resp.text
      retobj['data'] = dataset_contents
      retobj['fields'] = labels
      retobj['datatypes'] = dataset_typelist
      retobj['rows'] = dataset_row_count
      retobj['columns'] = dataset_column_count

      retobj['problem'] = generateSpecs.generateReturnedProblem(full_problem_spec)
      retobj['metadata'] = generateSpecs.generateMetadata(dataset_typelist,labels)
      retobj['dataset_schema'] = {}
      lastlabel = labels[-1:]
      retobj['yvar'] = lastlabel[0]
      print('returning dataset details:',retobj)
  

      return retobj
  
      #except:
        # return an empty file if there was a problem with the read
        #return retobj


    # this endpoint combines two datasets together.  It takes arguments for the two
    # girder fileIds to use and the type of join to perform. This doesn't currently authenticate to
    # girder, so publicly available files are assumed.  A Pandas dataframe is returned

    def _pairwiseDatasetMerge(fileId_1,  fileId_2, join_type="rows", join_column=None, right_join_column=None):

      # go get the datasets
      requesturl_1 = girder_api_prefix+"/file/"+fileId_1+"/download"
      requesturl_2 = girder_api_prefix+"/file/"+fileId_2+"/download"
      # build a merged name from the IDs of the source objects
      merge_name = 'merged_'+str(fileId_1)+'_'+str(fileId_2)

      try:
        # read the data in from the external girder instance
        resp_1 = requests.get(requesturl_1)
        resp_2 = requests.get(requesturl_1)
        data_1_str = resp_1.text
        data_2_str = resp_2.text
        # convert the two strings returned into lists of dictionaries
        data_as_file1 = StringIO(data_1_str)
        data_as_file2 = StringIO(data_2_str)
        data1_df = pd.read_csv(data_as_file1, sep=',')
        data2_df = pd.read_csv(data_as_file2, sep=',')

        if join_type == 'rows':
          result_df = pd.concat([data1_df,data2_df])
        else:
          # we allow the column names to be different if they are both specified, the default
          # is for both to be the same name, if one one column name is provided.
          if right_join_column==None:
            right_join_column = join_column
          result_df = pd.merge(data1_df, data2_df, how='inner', left_on=join_column, right_on=right_join_column)
        return (result_df, merge_name)
      except:
        print('error: a problem occurred receiving or merging datasets')
        return None      
      


    @access.public
    @autoDescribeRoute(
        Description('Combine two datasets via combining columns or rows')
         .param('fileId_1', 'the girder Id of the first file', required=True,paramType='query')
         .param('fileId_2', 'the girder Id of the second file', required=True,paramType='query')
         .param('join_type', 'specify either "rows" or "columns"', required=True,paramType='query')
         .param('join_column', 'the name of the dataset column to use for a side-by-side join', required=False,paramType='query')
 
    )
    def pairwiseDatasetMerge(self,params):
      self.requireParams('fileId_1', params)
      self.requireParams('join_type', params)
      self.requireParams('fileId_2', params)
      self.requireParams('join_column', params)
      fileId_1 = params['fileId_1']
      fileId_2 = params['fileId_2']
      join_type = params['join_type']
      (result_df, merged_name) = _pairwiseDatasetMerge(fileId_1,fileId_2,join_type=join_type, join_column=join_column)
      try:
          return result_df.to_csv(sep=',',index=False)
      except:
        print('error: a problem occurred receiving or merging datasets')
        return None


    # this endpoint accepts a GeoApp style exported data object and saves it as a new
    # dataset attached to a new item.  This is written as a simpler alternative
    # to streaming to girder's file/chunk API.  a dataType argument is accepted
    # to imply 
    @access.public
    @autoDescribeRoute(
        Description('Accept a serialized json object from geoapp and write as a new local file. ')
         .param('data', 'the object to be stored', required=True,paramType='query')
         .param('name', 'the name of the object to be stored', required=True,paramType='query')
         .param('dataType', 'geospatial, temporal, tabular, image', required=True, paramType='query')
    )
    def uploadGeoappDataset(self,params):
      self.requireParams('data', params)
      self.requireParams('dataType', params)
      self.requireParams('name', params)
      self.requireParams('size', params)
      dataString = params['data']
      dataType = params['dataType']
     
      if len(dataType)>0:
        name = params['dataType']+'_'+params['name']
      else:
        name = params['name']

      # convert the data to a CSV formatted string
      data = json.loads(dataString)
      #print(data)
      fields = data['fields']
      print('number of fields received:',len(fields))
      dataArray = data['data']
      print('number of elements received:',len(dataArray))
      # pack the values into a dictionary and convert to a dataframe
      dict_new = {}
      for idx in range(len(fields)):
        dict_new[fields[idx]] = np.transpose(dataArray)[idx]
      data_df = pd.DataFrame(dict_new)
      # now that we have a dataframe, clean out entries with 0 in lat or long
      if dataType == 'geospatial_trips':
        print(data_df.shape[0]," records before zero filter")
        # in USA, longitude is negative
        data_df = data_df.loc[data_df['pickup_longitude'] != 0.0]
        data_df = data_df.loc[data_df['pickup_latitude'] != 0.0]
        data_df = data_df.loc[data_df['dropoff_longitude'] != 0.0]
        data_df = data_df.loc[data_df['dropoff_latitude'] != 0.0]
      print(data_df.shape[0],' records after zero filter')

      # now open girder connection and stream the dataframe as a CSV stream
      # to create a CSV file item attached to a new girder item in the datasets/geospatial
      # folder.  Each invocation creates a new item with the date in the name
      gc = girder_client.GirderClient(apiUrl=girder_api_prefix)
      login = gc.authenticate('modsquad', 'd3md3m')
      dataString = data_df.to_csv(sep=',',index=False)
      stream = StringIO.StringIO(dataString)
      datasize = len(dataString)
      # get the folder ID for us to use
      folderRecord = gc.resourceLookup('collection/datasets/geospatial')
      folderID = folderRecord['_id']
      print('got folder ID:',folderID)
      resp = gc.uploadStreamToFolder(folderID, stream=stream, filename=name, size=datasize) 
      return resp


   # This endpoint joins two datasets and automatically saves the result back in
   # the local girder instance.  This endpoint wraps the merge algorithm presented previously.
    @access.public
    @autoDescribeRoute(
        Description('Combine two datasets via combining columns or rows and saving the results')
         .param('fileId_1', 'the girder Id of the first file', required=True,paramType='query')
         .param('fileId_2', 'the girder Id of the second file', required=True,paramType='query')
         .param('join_type', 'specify either "rows" or "columns"', required=True,paramType='query')
         .param('join_column', 'the name of the dataset column to use for a side-by-side join', required=False,paramType='query')
    )
    def pairwiseDatasetMergeWithStorage(self,params):
      self.requireParams('fileId_1', params)
      self.requireParams('join_type', params)
      self.requireParams('fileId_2', params)
      self.requireParams('join_column', params)
      fileId_1 = params['fileId_1']
      fileId_2 = params['fileId_2']
      join_type = params['join_type']
      join_column = params['join_column']
      (joined_df, merged_name) = _pairwiseDatasetMerge(fileId_1,fileId_2,join_type=join_type,join_column=join_column)
      dataString = joined_df.to_csv(sep=',',index=False)
      stream = StringIO.StringIO(dataString)
      datasize = len(dataString)
      # get the folder ID for us to use
      folderRecord = gc.resourceLookup('collection/datasets/geospatial')
      folderID = folderRecord['_id']
      print('got folder ID:',folderID)
      resp = gc.uploadStreamToFolder(folderID, stream=stream, filename=merged_name, size=datasize) 
      return resp




def load(info):
    info['apiRoot'].modsquad = Modsquad()
