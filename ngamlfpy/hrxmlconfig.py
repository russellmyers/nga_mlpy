"""
   HRX ML Config API web service related classes

"""

import json, requests
import unittest
from ngamlfpy.utils import find_in_list_partial


class MLModelConfig:
    '''

        This class provides a wrapper around access routines to HRX ML Config API web services,
        and provides an object representing the model configuration  for an ML Service for a customer

    '''

    #TODO - replace with HRX ML Config API Web Service URL
    URL_PREFIX = 'http://pseudotest.a2hosted.com/etp/'

    def __init__(self, j_config,selection=None):
        """
            Constructs model info object based on input JSON format j_config dict.

            Instance usually created via factory static method (eg ''MLModelConfig.get_model_config_from_web_service_for_cust''), which creates a j_config dict based on retrieval from hrx ML Config API web service

        """
        self.j_config = j_config

        if self.is_old_format():
            raise Exception('Error - old format configuration. Please convert to new format')

        if selection is None:
            self.selection = {}
        else:
            self.selection=selection


    def is_old_format(self):
        """*Deprecated.* Now expect all models in new ml framework format"""
        # check if old format model configuration

        if 'features' in self.j_config:
            if (self.j_config['features'] is None) or (len(self.j_config['features']) == 0):
                return True
        else:
            return True

        return False

    # def conv_pay_feature(self,feat):
    #     """*Deprecated. Now expect all models in new ml framework format"""
    #     new_feat = {}
    #     for field in feat:
    #         if field == 'sourcewagetype':
    #             new_feat['source'] = 'RT'
    #             new_feat['field'] = 'WT_' + feat[field]
    #         else:
    #             new_feat[field] = feat[field]
    #     new_feat['feature_type'] = 'N'
    #     return new_feat
    #
    # def conv_master_feature(self,feat):
    #     """*Deprecated. Now expect all models in new ml framework format"""
    #     new_feat = {}
    #     for field in feat:
    #         if field == 'sourceinfotype':
    #             new_feat['source'] = 'IT' + feat[field]
    #
    #         elif field == 'sourcefieldname':
    #             new_feat['field'] = feat[field]
    #         elif field == 'featuretype':
    #             new_feat['feature_type'] = feat[field]
    #         else:
    #             new_feat[field] = feat[field]
    #     return new_feat
    #
    # def conv_to_new_format(self):
    #     """*Deprecated. Now expect all models in new ml framework format"""
    #
    #     self.j_config['modelfeatures'] = []
    #     for pay_feat in self.j_config['modelpayfeatures']:
    #         self.j_config['modelfeatures'].append(self.conv_pay_feature(pay_feat))
    #
    #     for master_feat in self.j_config['modelmasterfeatures']:
    #         self.j_config['modelfeatures'].append(self.conv_master_feature(master_feat))
    #
    #
    #     if (self.j_config['modelselectionfields'] is None) or (len(self.j_config['modelselectionfields']) == 0):
    #         self.j_config['modelselectionfields'] = json.loads("""
    #         [
    #             {
    #               "source": "IT0001",
    #               "field": "PERNR",
    #               "title": "PERNR",
    #               "feature_type": "K",
    #               "data_type": "S",
    #               "conv": ""
    #             },
    #
    #
    #             {
    #               "source": "RT",
    #               "field": "For-period",
    #               "title": "ForPeriod",
    #               "feature_type": "K",
    #               "data_type": "S",
    #               "conv": ""
    #             }
    #           ]
    #         """)

    @staticmethod
    def call_web_service(endpoint, query_params=None, param=None, url_prefix=URL_PREFIX,headers=None):
        """General static method to call any of the endpoints within HRXMLConfig API.

           :param endpoint: Endpoint to call
           :type endpoint: str

           :param query_params: Extra query params to add to html url. Default None
           :type query_param: str

           :param url_prefix: Base url for endpoint. Defaults to location in "URL_PREFIX" constant

           :param headers: Optional parameter - headers to include headers for gcc and lcc
                            in web service call

           :returns: data,status code"""

        if query_params is None:
            if param is None:
                full_url = url_prefix + endpoint
            else:
                full_url = url_prefix + endpoint + '/' + param
        else:
            full_url = url_prefix + endpoint + query_params

        resp = requests.get(full_url)
        j_data = {}
        if resp.status_code != 200:
            print('Error reading from web service: ' + full_url + ' response status: ' + str(resp))
            return {}, resp.status_code

        j_data = json.loads(resp.text)
        print('web service call successful: ',endpoint, query_params)
        return j_data, resp.status_code


    @staticmethod
    def build_headers(gcc,lcc):
        """ Static method to build http header required by HRXMLConfig API"""
        return {'customerId': gcc,'companyId': lcc}

    @staticmethod
    def get_sample_customer_using_model(ml_service,model_name):
        """ Static method to find, for  a sample ml service, a  customer which utilises the model name specified.
        Returns ml service, gcc, lcc, variant, system of sample customer
        """

        headers = MLModelConfig.build_headers('ZZZ','Z99')
        endpoint = 'api/customer-models'
        query_params = ''

        j, status = MLModelConfig.call_web_service(endpoint, query_params=query_params, url_prefix=MLModelConfig.URL_PREFIX,headers=headers)

        cust_data = {}

        if status == 200:
            for item in j['Items']:
               # if (item['model_name'] == model_name) and (item['ml_service'] == ml_service):
                if (item['model'] == model_name):
                    endpoint = 'api/customer-models/' + str(item['id'])
                    j_group, status = MLModelConfig.call_web_service(endpoint, query_params=query_params,
                                                                          url_prefix=MLModelConfig.URL_PREFIX,
                                                                          headers=headers)
                    if (status == 200) and (j_group['mlService']['code'] == ml_service):
                        cust_data['gcc'] = j_group['customer']['gcc']
                        cust_data['lcc'] = j_group['customer']['lcc']
                        cust_data['variant'] = j_group['variant']
                        cust_data['system'] = j_group['systems'][0]
            if 'gcc'  not in cust_data:
                return None
            return cust_data
        else:
            return None

    @staticmethod
    def get_model_config_from_web_service_for_cust(ml_service: object, system: object, gcc: object, lcc: object, group: object = None,
                                                   payroll_area: object = None,
                                                   client: object = None,

                                                   endpoint: object = 'api/model-info',
                                                   url_prefix: object = URL_PREFIX,
                                                   use_headers: object = False) -> object:

        """Main static factory method to retrieve model config from model-info web service for gcc/lcc + (variant or payroll area) + system.
           Creates MLModelConfig object based on returned json, and returns the created object
        """

        query_params = '?ml_service=' + ml_service + '&system=' + system + '&gcc=' + gcc + '&lcc=' + lcc
        if group is None:
            pass
        else:
            query_params += '&variant=' + group

        if payroll_area is None:
            pass
        else:
            query_params += '&payroll_area=' + payroll_area

        headers = None
        if use_headers:
            headers = MLModelConfig.build_headers(gcc,lcc)

        j, status = MLModelConfig.call_web_service(endpoint, query_params=query_params, url_prefix=url_prefix, headers=headers)
        if status == 200:
            selection = {'gcc':gcc,'lcc':lcc,'ml_service':ml_service,'system':system,'variant':'None' if ((group is None) or (group == '')) else group,'payroll_area':'None' if ((payroll_area is None) or (payroll_area == '')) else payroll_area}
            return  MLModelConfig(j,selection=selection)
        else:
            return None

    @staticmethod
    def get_model_config_from_web_service_for_model(ml_service,model_name):
        """Static factory method to retrieve model config from web service for  model name.
           Finds, for an ml service, a customer which uses the model name specified, then retrieves model-info for that customer.
           Creates MLModelConfig object based on returned json
                """

        cust_data = MLModelConfig.get_sample_customer_using_model(ml_service, model_name)
        return MLModelConfig.get_model_config_from_web_service_for_cust(ml_service,
                                                                                     cust_data['system'],
                                                                                     cust_data['gcc'], cust_data['lcc'],
                                                                                     cust_data['variant'])

    @staticmethod
    def get_group_for_payroll_area(gcc,lcc,payroll_area,ml_service):
        """Static method to retrieve variant associated with specified payroll area for an ml service/gcc/lcc
        Returns variant code"""

        j_data,status = MLModelConfig.call_web_service('api/customer-models')

        if status == 200:
           for item in j_data['Items']:
                  j_group_data, status = MLModelConfig.call_web_service('api/customer-models/' + str(item['id']))
                  if status == 200:
                      if (j_group_data['customer']['gcc'] == gcc) and (j_group_data['customer']['lcc'] == lcc) and(j_group_data['mlService']['code']==ml_service) :
                          if payroll_area in j_group_data['payrollAreas']:
                             return j_group_data['variant']
                  else:
                      return ''

               #if (item['gcc'] == gcc) and (item['lcc'] == lcc) and (item['name'] == payroll_area):
               #   return item['group']
           return ''
        else:
           return ''

    @staticmethod
    def get_ml_service_details(ml_service):
        """ Static method to retrieve details for specified ml service from ml-services endpoint"""

        j_data,status = MLModelConfig.call_web_service('api/ml-services')

        if status == 200:
           for item in j_data['Items']:
                if (item['code'] == ml_service):
                    return item
           return {}
        else:
           return {}

    @staticmethod
    def is_unsupervised(ml_service):
        """ Static method to call ml-services endpoint in api and determine whether an ml service is supervised or unsupervised"""

        ml_service_details = MLModelConfig.get_ml_service_details(ml_service)
        if 'mlType' in ml_service_details:
            if ml_service_details['mlType'] == 'unsupervised':
                return True
            else:
                return False

        return False

    # instance methods:

    def get_model_name(self):
        """ Returns model name"""
        return self.j_config['modelCode']

    def get_model_version(self):
        """ Returns model version (zfilled to 3 chars)"""
        return self.j_config['modelVersion'].zfill(3)

    def get_all_fields(self):
        """ Returns  list containing all fields  used in model config
         (ie data source fields + selection fields + features + labels)"""

        return self.j_config['datasourceFields'] + self.j_config['selectionFields'] +  self.j_config['features'] + self.j_config['labels']


    def get_all_field_names(self):
        """ Returns list of all field titles used in model config
        (ie data source fields + selection fields + features + labels)"""

        all_fields = self.get_all_fields()
        field_names = [x['title'] for x in all_fields]

        return field_names

    def get_metadata(self):
        """ Returns metadata for model config """
        return {"id":self.j_config["modelCode"], "description":self.j_config["modelDescription"], "country": self.j_config["modelCountry"], "version": self.j_config["modelVersion"]}


    def get_feature_field_names(self):
        """ Returns list of field titles of all features used in model config """

        features =  self.get_feature_fields() #self.j_config['modelfeatures']
        field_names = [x['title'] for x in features]
        return field_names

    def get_feature_fields(self):
        """ Returns list containing all feature fields used in model config """

        features = self.j_config['features']
        return features


    def get_feature_field_names_with_type(self,feat_type):
        """
           Returns list containing all field titles for fields of specified feature_type used in model config.
             feature_types are:
                K: key fields (data source fields and selection fields)
                N: numeric features
                C: categorical features
                L: labels
        """

        all_fields = self.get_all_fields()

        field_names = [x['title'] for x in all_fields if x['feature_type'] == feat_type]
        return field_names

    # def get_selection_fields(self, cols):
    #
    #     feat_fields = self.get_all_feature_field_names()
    #
    #     feat_cols_with_cat = feat_fields[:]  # sel_fields + feat_fields
    #
    #     for feat_col in feat_fields:
    #         feat_cols_with_cat.extend(find_in_list_partial(feat_col, cols))
    #
    #     sel_fields = []
    #     for col in cols:
    #         if col in feat_cols_with_cat:
    #             pass
    #         else:
    #             sel_fields.append(col)
    #
    #     return sel_fields

    def get_selection_field_names(self):
        """ Returns list of field titles for all selection fields """

        field_names = [x['title'] for x in self.j_config['selectionFields']]
        return field_names


    def get_data_source_field_names(self):
        """ Returns list of field titles for all data source fields """

        field_names = [x['title'] for x in self.j_config['datasourceFields']]
        return field_names


    def get_field_with_title(self,title):
        all_fields = self.get_all_fields()
        for field in all_fields:
            if field['title'] == title:
                return field

        return None

    def get_exclusions(self):
        """ returns list of exclusion sets """

        exclusions = []
        for excl_record in self.j_config['exclusions']:
            curr_exclusion_set = []
            for excl_item in excl_record:
                curr_exclusion_set.append({'infotype': excl_item['sourceinfotype'],'field': excl_item['sourcefieldname'], 'value': excl_item['exclusionvalue']})

            exclusions.append(curr_exclusion_set)

        return exclusions

    def get_label_col(self):
        """ Returns field title of label (target) field """

        out = None

        all_fields = self.get_all_fields()

        for feature in all_fields:
            if feature['feature_type'] == 'L':
                out = feature['title']
                return out

        return out


    def is_unlabelled(self):
        """ Checks if this model is unlabelled (ie has no label field) """

        label = self.get_label_col()

        if label is None:
            return False
        else:
            return True

    def get_title_for_field(self,field_name):
        """ Returns field title for specified field source name.
         Note: There may be multiple fields with same field source name (eg when conversions also apply). In this case,
         returns title of field with  specified field source name and no conversion (ie "raw" field)
         """

        all_fields = self.get_all_fields()

        candidates = []
        for field in all_fields:
            if field['field'] == field_name:
               candidates.append(field)

        if len(candidates) == 0:
            return None

        if len(candidates) > 1:
           for field in candidates:
               if field['conv'] == '':
                   return field['title']
           return None
        else:
            return candidates[0]['title']

        return None

    # def all_rt_fields(self, excl_wts=False):
    #     out = []
    #     all_fields = self.get_all_fields()
    #
    #     for feature in all_fields:
    #         if feature['source'][:2] == 'RT':
    #             if (excl_wts and feature['field'][:3] == 'WT_'):
    #                 pass
    #             else:
    #                 out.append(feature['field'])
    #
    #     return out

    def rt_wt_features(self):
        """ Returns list of payment code features """

        out = []

        all_fields = self.get_feature_fields()

        for feature in all_fields:
            if feature['source'][:2] == 'RT':
                if (feature['field'][:3] == 'WT_'):
                    out.append(feature)

        return out

    # def master_fields_per_infotype(self):
    #
    #     out = {}
    #
    #     all_fields = self.get_all_fields()
    #     for feature in all_fields:
    #         if feature['source'][:2] == 'IT':
    #             if feature['source'] in out:
    #                 out[feature['source']].append(feature['field'])
    #             else:
    #                 out[feature['source']] = [feature['field']]
    #
    #     return out

    def get_categorical_features(self):
        """ Returns list of field titles for categorical features """

        out = []
        all_fields = self.get_all_fields()

        for feature in all_fields:
            if feature['feature_type'] == 'C':
                out.append(feature['title'])

        return out

    # def get_selection_field_names_with_curr_cols(self, cols):
    #
    #     feat_fields = self.get_all_feature_field_names()
    #
    #     feat_cols_with_cat = feat_fields[:]  # sel_fields + feat_fields
    #
    #     for feat_col in feat_fields:
    #         feat_cols_with_cat.extend(find_in_list_partial(feat_col, cols))
    #
    #     sel_fields = []
    #     for col in cols:
    #         if col in feat_cols_with_cat:
    #             pass
    #         else:
    #             sel_fields.append(col)
    #
    #     return sel_fields

    # def get_key_fields(self):
    #     out = []
    #
    #     all_fields = self.get_all_fields()
    #
    #     for feature in all_fields:
    #         if feature['feature_type'] == 'K':
    #             out.append(feature['title'])
    #
    #     return out

    # def get_num_sel_and_key_cols(self, cols):
    #     """" For specified list of columns, Determine how many relate to """
    #     num_sel_cols = len(self.get_selection_fields(cols))
    #     num_key_cols = len(self.get_key_fields())
    #     return num_sel_cols + num_key_cols

    def assemble_file_prefix(self, period, multi=False, cust=None, data_dir='data', extra_dir=None):
        """ Used for testing purposes """

        if multi:
            file_prefix =  self.get_metadata()['id'] #self.j_config['model_metadata']['model']
            # file_prefix = '_'.join([
            #     j['model_metadata']['model'],
            #     target_vers
            # ])
        elif cust is None:
            file_prefix = '_'.join([
                self.get_metadata()['id'],
                # target_vers,
                self.selection['system'],
                self.selection['gcc'],
                self.selection['lcc'],
                self.selection['variant'],
                period
            ])
        else:
            file_prefix = '_'.join([
                self.get_metadata()['id'],
                # target_vers,
                cust['system'],
                cust['client'],
                cust['gcc'],
                cust['lcc'],
                cust['payroll_area'],
                period
            ])

        if extra_dir is None:
            pass
        else:
            file_prefix = '/'.join([extra_dir, file_prefix])

        # file_prefix = '/'.join([j['model_metadata']['model'] + '_' + target_vers,file_prefix])
        file_prefix = '/'.join([self.get_metadata()['id'], file_prefix])

        file_prefix = '/'.join([data_dir, file_prefix])
        return file_prefix

    def summarise(self):
        """ Print summary of model config """

        print('selection: ' + str(self.selection))
        print('Model metadata: ' + str(self.get_metadata()))
        print('Num emp fields: ' + str(len(self.get_all_field_names())))
        print('..Num cat features: ' + str(len(self.get_feature_field_names_with_type('C'))))
        print('..Num numeric features: ' + str(len(self.get_feature_field_names_with_type('N'))))
        print('..Num key fields: ' + str(len(self.get_feature_field_names_with_type('K'))))
        print('..Num label fields: ' + str(len(self.get_feature_field_names_with_type('L'))) + ' ' + str(self.get_feature_field_names_with_type('L')))



# def get_pay_data_from_web_service(ml_service,system,client,gcc,lcc,payroll_area,period,url_prefix='http://pseudotest.a2hosted.com/etp/pull_pay_data'):
#     full_url = url_prefix + '?ml_service=' + ml_service + '&system=' + system + '&client=' + client + '&gcc=' + gcc + '&lcc=' + lcc + '&payroll_area=' + payroll_area + '&period=' + period
#
#     resp = requests.get(full_url)
#     j_data = {}
#     if resp.status_code != 200:
#         print('Error reading from web service: ' + str(resp))
#         return {},resp.status_code
#
#     j_data = json.loads(resp.text)
#     print(  str(j_data['selection']) + ' Num emp records read: ' + str(len(j_data['values'])))
#
#     return j_data,resp.status_code




# class VariantFinder:
#
#     def __init__(self):
#         pass
#
#     @staticmethod
#     def variant_to_payroll_area(ml_service,gcc,lcc,variant):
#
#         if (ml_service == 'TWV') and (gcc == 'ZCS') and (lcc == 'Z10'):
#             if variant is None:
#                 return 'X1'
#             elif variant == 'USWK':
#                return 'X3'
#             elif variant == 'USBWK':
#                return 'X2'
#             elif variant == 'USMTH':
#                 return 'X1'
#             elif variant == '':
#                return 'X1'
#             else:
#                return None
#
#         return None



class TestModelConfig(unittest.TestCase):

    def test_read_by_payroll_area(self):
        ml_model_config_cust = MLModelConfig.get_model_config_from_web_service_for_cust('TWV', 'EQ1', 'ZCS', 'Z10',
                                                                                        payroll_area='X1')
        self.assertIsNotNone(ml_model_config_cust,"model config returned none")

    def test_read_by_group(self):
        ml_model_config_cust = MLModelConfig.get_model_config_from_web_service_for_cust('TWV', 'EQ1', 'ZCS', 'Z10',
                                                                                        group='MTHLY')
        self.assertIsNotNone(ml_model_config_cust, "model config returned none")

    def test_non_existing_payroll_area(self):
        ml_model_config_cust = MLModelConfig.get_model_config_from_web_service_for_cust('TWV', 'EQ1', 'ZCS', 'Z10',
                                                                                        payroll_area='XQ')
        self.assertIsNone(ml_model_config_cust, "model config should not have returned anything")

    def test_non_existing_group(self):
        ml_model_config_cust = MLModelConfig.get_model_config_from_web_service_for_cust('TWV', 'EQ1', 'ZCS', 'Z10',
                                                                                        group = 'QQQQ')
        self.assertIsNone(ml_model_config_cust, "model config should not have returned anything")

    def test_no_group_or_payroll_area_supplied(self):
        ml_model_config_cust = MLModelConfig.get_model_config_from_web_service_for_cust('TWV', 'EQ1', 'ZCS', 'Z10')
        self.assertIsNone(ml_model_config_cust, "model config should not have returned anything")


if __name__ == '__main__':

    model = 'T001'
    run_unit_tests = True

    if run_unit_tests:
        unittest.main()

    ml_model_config_cust = MLModelConfig.get_model_config_from_web_service_for_cust('TWV','EQ1','ZCS','Z10',group='MTHLY')

    ml_model_config_cust = MLModelConfig.get_model_config_from_web_service_for_cust('TWV', 'EQ1', 'ZCS', 'Z10',payroll_area='X1')

    meta = ml_model_config_cust.get_metadata()
    feature_field_names = ml_model_config_cust.get_all_feature_field_names()
    feature_field_names_labels = ml_model_config_cust.get_feature_field_names_with_type('L')
    feature_field_names_numeric = ml_model_config_cust.get_feature_field_names_with_type('N')
    feature_label = ml_model_config_cust.get_label_col()
    rt_fields = ml_model_config_cust.rt_fields()
    rt_wts = ml_model_config_cust.rt_wts()
    master_fields_per_infotype = ml_model_config_cust.master_fields_per_infotype()
    cat_feats = ml_model_config_cust.get_categorical_features()
    #sel_fields = ml_model_config_cust.get_selection_fields()
    key_fields = ml_model_config_cust.get_key_fields()
    #num_sel_and_key_cols = ml_model_config_cust.get_num_sel_and_key_cols()
    pref = ml_model_config_cust.assemble_file_prefix('99X23',multi=True,cust=None)
    sum = ml_model_config_cust.summarise()



    # model_config_cust = ModelConfig.get_model_config_from_web_service_for_cust('TWV','EQ1','402','ZCS','Z04','E1')
    # if model_config_cust:
    #      model_config_cust.summarise()
    # else:
    #     print('cust - fail')
    #
    # model_config_model = ModelConfig.get_model_config_from_web_service_for_model(model)
    # if model_config_model:
    #     model_config_model.summarise()
    # else:
    #     print('model - fail')


    print('complete')

