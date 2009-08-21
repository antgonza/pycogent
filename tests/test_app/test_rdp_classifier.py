#!/usr/bin/env python
"""Tests for the rdp_classifier_2.0.1 application controller"""

from os import getcwd, environ, remove
from shutil import rmtree
from cogent.app.util import ApplicationNotFoundError, ApplicationError,\
    get_tmp_filename
from cogent.app.rdp_classifier import RdpClassifier, assign_taxonomy
from cogent.util.unit_test import TestCase, main

__author__ = "Kyle Bittinger"
__copyright__ = "Copyright 2007-2009, The Cogent Project"
__credits__ = ["Kyle Bittinger"]
__license__ = "GPL"
__version__ = "1.4.0.dev"
__maintainer__ = "Kyle Bittinger"
__email__ = "kylebittinger@gmail.com"
__status__ = "Prototype"

class RdpClassifierTests(TestCase):
    def setUp(self):
        # fetch user's RDP_JAR_PATH
        if 'RDP_JAR_PATH' in environ:
            self.user_rdp_jar_path = environ['RDP_JAR_PATH']
        else:
            self.user_rdp_jar_path = 'rdp_classifier-2.0.jar'

    def test_default_java_vm_parameters(self):
        """RdpClassifier should store default arguments to Java VM."""
        a = RdpClassifier()
        self.assertContains(a.Parameters, '-Xmx')
        self.assertEqual(a.Parameters['-Xmx'].Value, '1000m')

    def test_parameters_list(self):
        a = RdpClassifier()
        parameters = a.Parameters.keys()
        parameters.sort()
        self.assertEqual(parameters, ['-Xmx', '-training-data'])

    def test_jvm_parameters_list(self):
        a = RdpClassifier()
        parameters = a.JvmParameters.keys()
        parameters.sort()
        self.assertEqual(parameters, ['-Xmx'])

    def test_positional_parameters_list(self):
        a = RdpClassifier()
        parameters = a.PositionalParameters.keys()
        parameters.sort()
        self.assertEqual(parameters, ['-training-data'])

    def test_default_positional_parameters(self):
        """RdpClassifier should store default positional arguments."""
        a = RdpClassifier()
        self.assertContains(a.PositionalParameters, '-training-data')
        self.assertEqual(a.PositionalParameters['-training-data'].Value, '')        

    def test_assign_jvm_parameters(self):
        """RdpCalssifier should pass alternate parameters to Java VM."""
        app = RdpClassifier()
        app.Parameters['-Xmx'].on('75M')
        exp = ''.join(['cd "', getcwd(), '/"; java -Xmx75M -jar "', self.user_rdp_jar_path, '"'])
        self.assertEqual(app.BaseCommand, exp)

    def test_basecommand_property(self):
        """RdpClassifier BaseCommand property should use overridden _get_base_command method."""
        app = RdpClassifier()
        self.assertEqual(app.BaseCommand, app._get_base_command())

    def test_base_command(self):
        """RdpClassifier should return expected shell command."""
        app = RdpClassifier()
        exp = ''.join(['cd "', getcwd(), '/"; java -Xmx1000m -jar "', self.user_rdp_jar_path, '"'])
        self.assertEqual(app.BaseCommand, exp)
        
    def test_change_working_dir(self):
        """RdpClassifier should run program in expected working directory."""
        test_dir = '/tmp/RdpTest'

        app = RdpClassifier(WorkingDir=test_dir)
        exp = ''.join(['cd "', test_dir, '/"; java -Xmx1000m -jar "', self.user_rdp_jar_path, '"'])
        self.assertEqual(app.BaseCommand, exp)

        rmtree(test_dir)

    def test_sample_fasta(self):
        """RdpClassifier should classify its own sample data correctly"""
        test_dir = '/tmp/RdpTest'
        app = RdpClassifier(WorkingDir=test_dir)

        results_file = app(rdp_sample_fasta)['StdOut']
        
        id_line = results_file.readline()
        self.failUnless(id_line.startswith('>X67228'))

        classification_line = results_file.readline().strip()
        obs = parse_rdp(classification_line)
        exp = ['Root', 'Bacteria', 'Proteobacteria', 'Alphaproteobacteria', 'Rhizobiales', 'Rhizobiaceae', 'Rhizobium']
        self.assertEqual(obs, exp)

        rmtree(test_dir)

    def test_custom_training_data(self):
        """RdpClassifier should look for custom training data"""
        test_dir = '/tmp/RdpTest'
        app = RdpClassifier(WorkingDir=test_dir)

        nonexistent_training_data = get_tmp_filename(
            prefix='RdpTestCustomTrainingData', suffix='.properties')
        app.Parameters['-training-data'].on(nonexistent_training_data)
        self.assertRaises(ApplicationError, app, rdp_sample_fasta)
        
        rmtree(test_dir)

class RdpWrapperTests(TestCase):
    """ Tests of RDP classifier wrapper functions
    """
    def setUp(self):
        self.test_input1 = rdp_test_fasta.split('\n')
        self.expected_assignments1 = rdp_expected_out
        
    def test_assign_taxonomy(self):
        """assign_taxonomy wrapper functions as expected 
        
            This test may fail periodicially, but failure should be rare.
        
        """
        # convert the expected dict to a list, so it's easier to 
        # handle the order
        expected_assignments = \
         [(k,v[0],v[1]) for k,v in self.expected_assignments1.items()]
        expected_assignments.sort()
        
        # Because there is some variation in the taxon assignments, 
        # I run the test several times (which can be quite slow) and 
        # each sequence was classified the same as expected at least once
        taxon_assignment_results = [False] * len(expected_assignments)
        all_assigned_correctly = False
        for i in range(10):
            actual_assignments = assign_taxonomy(self.test_input1)
            # covert actual_assignments to a list so it's easier to handle
            # the order
            actual_assignments = \
             [(k,v[0],v[1]) for k,v in actual_assignments.items()]
            actual_assignments.sort()
            
            for j in range(len(expected_assignments)):
                a = actual_assignments[j]
                e = expected_assignments[j]
                # same description fields
                self.assertEqual(a[0],e[0])
                
                # same taxonomic assignment
                r = a[1] == e[1]
                if r and not taxon_assignment_results[j]:
                    taxon_assignment_results[j] = True
                
                # confidence >= 0.80
                self.assertTrue(a[2]>=0.80)
                
            if False not in taxon_assignment_results:
                # all sequences have been correctly assigned at
                # least once -- bail out
                all_assigned_correctly = True
                break
        
        # make sure all taxonomic results were correct at least once
        self.assertTrue(all_assigned_correctly)
            
    def test_assign_taxonomy_alt_confidence(self):
        """assign_taxonomy wrapper functions as expected with alt confidence
        """
        actual_assignments = \
         assign_taxonomy(self.test_input1,min_confidence=0.95)            
        # covert actual_assignments to a list so it's easier to handle
        # the order
        actual_assignments = \
             [(k,v[0],v[1]) for k,v in actual_assignments.items()]
        actual_assignments.sort()
        
        # convert the expected dict to a list, so it's easier to 
        # handle the order
        expected_assignments = \
         [(k,v[0],v[1]) for k,v in self.expected_assignments1.items()]
        expected_assignments.sort()
        
        for a,e in zip(actual_assignments,expected_assignments):
            # same description fields
            self.assertEqual(a[0],e[0])
            # confidence >= 0.95
            self.assertTrue(a[2]>=0.95)
            
    def test_assign_taxonomy_file_output(self):
        """ assign_taxonomy wrapper writes correct file output when requested
        
            This function tests for sucessful completion of assign_taxonomy
             when writing to file, that the lines in the file roughly look
             correct by verifying how many are written (by zipping with 
             expected), and that each line starts with the correct seq id.
             Actual testing of taxonomy data is performed elsewhere.
        
        """
        output_fp = get_tmp_filename(\
         prefix='RDPAssignTaxonomyTests',suffix='.txt')
        # convert the expected dict to a list of lines to match 
        # file output
        expected_file_headers = self.expected_assignments1.keys()
        expected_file_headers.sort()
        
        actual_return_value = assign_taxonomy(\
         self.test_input1,min_confidence=0.95,output_fp=output_fp)
        
        actual_file_output = list(open(output_fp))
        actual_file_output.sort()

        # remove the output_fp before running the tests, so if they
        # fail the output file is still cleaned-up
        remove(output_fp)
        
        # None return value on write to file
        self.assertEqual(actual_return_value,None)
        
        # check that each line starts with the correct seq_id -- not 
        # checking the taxonomies or confidences here as these are variable and
        # tested elsewhere
        for a,e in zip(actual_file_output,expected_file_headers):
            self.assertTrue(a.startswith(e))
        
            
def parse_rdp(line):
    """Returns a list of assigned taxa from an RDP classification line
    """
    tokens = line.split('; ')
    # Keep even-numbered tokens
    return [t for (pos, t) in enumerate(tokens) if (pos % 2 == 0)]

# Sample data copied from rdp_classifier-2.0, which is licensed under
# the GPL 2.0 and Copyright 2008 Michigan State University Board of
# Trustees

rdp_sample_fasta = """>X67228 Bacteria;Proteobacteria;Alphaproteobacteria;Rhizobiales;Rhizobiaceae;Rhizobium
aacgaacgctggcggcaggcttaacacatgcaagtcgaacgctccgcaaggagagtggcagacgggtgagtaacgcgtgggaatctacccaaccctgcggaatagctctgggaaactggaattaataccgcatacgccctacgggggaaagatttatcggggatggatgagcccgcgttggattagctagttggtggggtaaaggcctaccaaggcgacgatccatagctggtctgagaggatgatcagccacattgggactgagacacggcccaaa
"""
rdp_sample_classification = """>X67228 reverse=false
Root; 1.0; Bacteria; 1.0; Proteobacteria; 1.0; Alphaproteobacteria; 1.0; Rhizobiales; 1.0; Rhizobiaceae; 1.0; Rhizobium; 0.95; 
"""

rdp_test_fasta = """>AY800210 description field
TTCCGGTTGATCCTGCCGGACCCGACTGCTATCCGGATGCGACTAAGCCATGCTAGTCTAACGGATCTTCGGATCCGTGGCATACCGCTCTGTAACACGTAGATAACCTACCCTGAGGTCGGGGAAACTCCCGGGAAACTGGGCCTAATCCCCGATAGATAATTTGTACTGGAATGTCTTTTTATTGAAACCTCCGAGGCCTCAGGATGGGTCTGCGCCAGATTATGGTCGTAGGTGGGGTAACGGCCCACCTAGCCTTTGATCTGTACCGGACATGAGAGTGTGTGCCGGGAGATGGCCACTGAGACAAGGGGCCAGGCCCTACGGGGCGCAGCAGGCGCGAAAACTTCACAATGCCCGCAAGGGTGATGAGGGTATCCGAGTGCTACCTTAGCCGGTAGCTTTTATTCAGTGTAAATAGCTAGATGAATAAGGGGAGGGCAAGGCTGGTGCCAGCCGCCGCGGTAAAACCAGCTCCCGAGTGGTCGGGATTTTTATTGGGCCTAAAGCGTCCGTAGCCGGGCGTGCAAGTCATTGGTTAAATATCGGGTCTTAAGCCCGAACCTGCTAGTGATACTACACGCCTTGGGACCGGAAGAGGCAAATGGTACGTTGAGGGTAGGGGTGAAATCCTGTAATCCCCAACGGACCACCGGTGGCGAAGCTTGTTCAGTCATGAACAACTCTACACAAGGCGATTTGCTGGGACGGATCCGACGGTGAGGGACGAAACCCAGGGGAGCGAGCGGGATTAGATACCCCGGTAGTCCTGGGCGTAAACGATGCGAACTAGGTGTTGGCGGAGCCACGAGCTCTGTCGGTGCCGAAGCGAAGGCGTTAAGTTCGCCGCCAGGGGAGTACGGCCGCAAGGCTGAAACTTAAAGGAATTGGCGGGGGAGCAC
>EU883771
TGGCGTACGGCTCAGTAACACGTGGATAACTTACCCTTAGGACTGGGATAACTCTGGGAAACTGGGGATAATACTGGATATTAGGCTATGCCTGGAATGGTTTGCCTTTGAAATGTTTTTTTTCGCCTAAGGATAGGTCTGCGGCTGATTAGGTCGTTGGTGGGGTAATGGCCCACCAAGCCGATGATCGGTACGGGTTGTGAGAGCAAGGGCCCGGAGATGGAACCTGAGACAAGGTTCCAGACCCTACGGGGTGCAGCAGGCGCGAAACCTCCGCAATGTACGAAAGTGCGACGGGGGGATCCCAAGTGTTATGCTTTTTTGTATGACTTTTCATTAGTGTAAAAAGCTTTTAGAATAAGAGCTGGGCAAGACCGGTGCCAGCCGCCGCGGTAACACCGGCAGCTCGAGTGGTGACCACTTTTATTGGGCTTAAAGCGTTCGTAGCTTGATTTTTAAGTCTCTTGGGAAATCTCACGGCTTAACTGTGAGGCGTCTAAGAGATACTGGGAATCTAGGGACCGGGAGAGGTAAGAGGTACTTCAGGGGTAGAAGTGAAATTCTGTAATCCTTGAGGGACCACCGATGGCGAAGGCATCTTACCAGAACGGCTTCGACAGTGAGGAACGAAAGCTGGGGGAGCGAACGGGATTAGATACCCCGGTAGTCCCAGCCGTAAACTATGCGCGTTAGGTGTGCCTGTAACTACGAGTTACCGGGGTGCCGAAGTGAAAACGTGAAACGTGCCGCCTGGGAAGTACGGTCGCAAGGCTGAAACTTAAAGGAATTGGCGGGGGAGCACCACAACGGGTGGAGCCTGCGGTTTAATTGGACTCAACGCCGGGCAGCTCACCGGATAGGACAGCGGAATGATAGCCGGGCTGAAGACCTTGCTTGACCAGCTGAGA
>EF503699
AAGAATGGGGATAGCATGCGAGTCACGCCGCAATGTGTGGCATACGGCTCAGTAACACGTAGTCAACATGCCCAGAGGACGTGGACACCTCGGGAAACTGAGGATAAACCGCGATAGGCCACTACTTCTGGAATGAGCCATGACCCAAATCTATATGGCCTTTGGATTGGACTGCGGCCGATCAGGCTGTTGGTGAGGTAATGGCCCACCAAACCTGTAACCGGTACGGGCTTTGAGAGAAGGAGCCCGGAGATGGGCACTGAGACAAGGGCCCAGGCCCTATGGGGCGCAGCAGGCACGAAACCTCTGCAATAGGCGAAAGCTTGACAGGGTTACTCTGAGTGATGCCCGCTAAGGGTATCTTTTGGCACCTCTAAAAATGGTGCAGAATAAGGGGTGGGCAAGTCTGGTGTCAGCCGCCGCGGTAATACCAGCACCCCGAGTTGTCGGGACGATTATTGGGCCTAAAGCATCCGTAGCCTGTTCTGCAAGTCCTCCGTTAAATCCACCCGCTTAACGGATGGGCTGCGGAGGATACTGCAGAGCTAGGAGGCGGGAGAGGCAAACGGTACTCAGTGGGTAGGGGTAAAATCCTTTGATCTACTGAAGACCACCAGTGGTGAAGGCGGTTCGCCAGAACGCGCTCGAACGGTGAGGATGAAAGCTGGGGGAGCAAACCGGAATAGATACCCGAGTAATCCCAACTGTAAACGATGGCAACTCGGGGATGGGTTGGCCTCCAACCAACCCCATGGCCGCAGGGAAGCCGTTTAGCTCTCCCGCCTGGGGAATACGGTCCGCAGAATTGAACCTTAAAGGAATTTGGCGGGGAACCCCCACAAGGGGGAAAACCGTGCGGTTCAATTGGAATCCACCCCCCGGAAACTTTACCCGGGCGCG
>random_seq
AAGCTCCGTCGCGTGAGCTAAAAACCATGCTGACTTATGAGACCTAAAAGCGATGCGCCGACCTGACGATGCTCTGTTCAGTTTCATCACGATCACCGGTAGTCAGGGTACCCTCCAGACCGCGCATAGTGACTATGTTCCCGCACCTGTATATGTAATTCCCATTATACGTCTACGTTATGTAGTAAAGTTGCTCACGCCAGGCACAGTTTGTCTTGATACATAGGGTAGCTTAAGTCCCGTCCATTTCACCGCGATTGTAATAGACGAATCAGCAGTGGTGCAATCAAGTCCCAACAGTTATATTTCAAAAATCTTCCGATAGTCGTGGGCGAAGTTGTCAACCTACCTACCATGGCTATAAGGCCCAGTTTACTTCAGTTGAACGTGACGGTAACCCTACTGAGTGCACGATACCTGCTCAACAACGGCCCAAAACCCGTGCGACACATTGGGCACTACAATAATCTTAGAGGACCATGGATCTGGTGGGTGGACTGAAGCATATCCCAAAAGTGTCGTGAGTCCGTTATGCAATTGACTGAAACAGCCGTACCAGAGTTCGGATGACCTCTGGGTTGCTGCGGTACACACCCGGGTGCGGCTTCTGAAATAGAAAAGACTAAGCATCGGCCGCCTCACACTTCAAGGGCCCTATGCCTAACAGTCTAGCAAATGCTTGAACCTTGTACCAAAGTTCAGACTTACCTTTACTTGGTTATCGCCCTTGAACCTGTAACCGTCACGTGGTCTACAATTCGTGGATTCCTGCATGAGGATGAACGGGTCCCTTCTCGTGCTACGTAGGCAGTATGTTCAACAACAAGAGGGTAATGCAATGGGGCTTAGAATCCCTTGACCGCAGAACACGTGGGGACTGCATTTCCACGTCGGTACTAATCCTCTGATTCTGTTCGTTATGTCCAAGACCATATCATCTTAATACCTATCAGACCTACGCTCTTCTGTCTGAACCCTAAAACGTTCGGAATAGCACGATCGTCCGGTTTGAACTGACGTGTAGGACTCTCGTTGCTCCCTGGTAATCATTCAGGCCCTGGCGTAATGCCCTTACTGACTACCACTGGCCTGGTTAGCAGCATCATTCTTAGGTGGAAGTGTTGGTCGCTGTTCTCCCATTGTGTCGAGTTTCCTCCGCGATCTCGTCACCGGGCGAGTTTTTAACTGCGACATCAACGTGTGGGTTTGACTTACCGCGCAGCATCGGGCCAACAGCCTTCGACAGCTAGAAGCTGGAAAATTAGTAAATAGTCCAGCACCGTTATATACATCTACTTGCTCGGAGGGGCACAGACGGCAGCGGACTCACGCCCGTTTTCAACCCTGGTAAAGAGGACGCGCGTTCATGAGGCGTAGGGATGCATTGCTCCTGATCCATGTTCTCAATTACGCTTCAATTCTGAGTATTAGACCTCACGCACGCAGTAAGCTTACTGGTCTCACTGCCATATTACTAGCTTAAAATAGCGGTCCAAAGACGCGGCATGATTAACGACAGCCCTTCACTTCACGGACCTCACGCCCGATATAGCGTACGTTCAGGGCTTGGAAAGGCGGACAATAATAAGCATGCAATAGCTAAGATTGTAGGAACTTCCATCCGCCGAACACAGCCCCTCGCCACAACGGTTGGAACCCGCGCTATGCCGTAAAGCGGCCAAAACGTCGCGCGCCACCACAAGATAGTGCTCAAAGCCCGCAGGGAGAGTCGGTGCTTGGTGCCTTCGCTACGGGGCCCAATAGCTTGCTTTTTCTTGCGCCGATTGACTCTAGGTAAACTCACCGTGACATACCGCATAAATCTGCAAGGGTGGTCCTGACTAAAGAGCTCTATAGCGATGATGGTGGCCTTAGACAGCACAAGCTGAACTTATTAATTCTTACCAGGTCCCGTGGGGTTCCGGAACATGAGTATGCTCTTTGGAACGGGCTTTGTCCACGTGTAAGGATGGTTACGTCCCGGGCGTATTGCACTACGTTCAAGTGGTGTAAGACAGAGTGACTAGAGACAGTGCCGTTATCACTATTGTGGGGCCCATCCTAAGGCTGAGGACACGGAACATGTCTCTTTATCATCGCACGAGCTGTATGCCACTGTATTCCTCTACCTTAGCCATCGCTCATTTAACGCCACGTGTAGCGGTGCAGGCTACGAGGCTTAAGTCTGCTCGGCTTGCTGAGCATATCCCTATAAATGAATGAGGAATAAAGGACATGACGCATTCCCGGCACCTGACAACAGGACGCAATTACTAAGTAGGCTTATGTAGTCTCGTGTAATGCAGACCGCTCTTAAGAGTCGGATCATAATTTGAGCAGAACAAAATTACTTATGCCCTTACAAGACGTGTGCAAACCTAAGTGTGAAGATTTAGGAGGCACCGCGTTTTATGGCTTCTGCGAATATATTGTGATTTCCTGAATAGTGGGGTGGGATGTAATGGACTGAAAAAGGTGAACATCTTAAGCCTACCAGTCATATTCCCGCCGGAACTTACTTAAATCAATGGACACTCAAAGAGACTTTGAAGCTCTTTATACCGATGTGCGCGCAAACCCCTCGAGTGCTGTCCTGCAACACCCTAAGTTGCAGTATGTTGGTTACAGCGATCATTTATAGGTTAAAATGGCTAAATGAGCTAGCGCCGCGGCCCGGAAGAACAGATTTCGTGAGGTGACCTGGCGAATGTGAATCCTGAAAATTTTCACACGACCACAGCAAAGCCGTTGGGCAAGTTCCGGCAACTAAGCCGAAGGTGACCCTGTACTGGCAGGGGTTTACCATGATGGAATGACCGGAATCAAGGCAAGGAAACACCGAGTACAGTTACGAGCAAACACGCGTAAATTAAATTACTGCAGTATAACTCGTTCACCAATTCGGGTCGGCCGACGTGCACGTCAACCAGGCATACGACACGTGAAATTCACTGCCGACACCACTGTTTCGATTACCATGTGTCTGGTCTTTCAAAGCACAGAGAGAGGCCCTCGCCGGTAAATTACGGACTCGTATGTGTTAACCGGGAATAGGTGGGACGGATCAACACTTAAGTTGGACAGACCAAAAACTAGCCGAAAACCTTCACAAAAAGAGTCATGAATGATCCGTCAAGGACAGCGCTCTCCCACCGCTGGATGGCAATCGCAAGGTATCAATCAATAGTGATGTCTACGAGTCTTACACAGGTGTCCTGGATGTAACTACTGTTGCCACGAGGAACGTATACACCCCAGCCTGCGTAATGTATGATCTTTTCGCGTTCTGAATCCAGAATATATTAGTGAAGGTCCCAAAGACACCTATTAATCGGTCCCAGTCGTTTGTCCTACATTTCTGTGTAGCCAGGGGTCCCCTATTACTCCTAATGAGGGATTGGCGCCGGGTAGAGTCTACGCCAAAGCGGCAAAGAGATTGATCACCGGCCGGGTACAATGCAGACTTTACATTCAGAGCGTGTTTGCCGCGTAATGCTTAATTCACTGCTGGCGCACCTCGCAGAATTACCTATATTTCCTCTTCCTCACTAAACTGGTGTTCAGAGATGGTCGATTTTCCGGTGTGGTTTATAGCAGGTCCCGCCACATGCAACACATACCGAACATCGCTATCAGTGTTGTTCTCGTCGCCGCGACTCCTGACTCGCGATTAGTTGGCTAGCTCCCAGCGCTAGCTCCGCCTCTGTCTGTATATCGCCAGTAATCAGTTTTCAATGACGTTGACTTATTTATAAACGAGCTGAAGCATCTCTCGCGCCCTAGCGTTACTACTATCAGGAACCGCCGTGTGGGAACTCCTCTACCTCACCACCCTGCCAGCTCCTATGACAACGTTTAGTCCGCGTACTGACAGGGAGAGAGAAGCGTTACGGGACCGTCTGAACAGTATGTTGGCAGAGGAAGGCCAGGGCTCCCTATGGTTTTAGGTT
>DQ260310
GATACCCCCGGAAACTGGGGATTATACCGGATATGTGGGGCTGCCTGGAATGGTACCTCATTGAAATGCTCCCGCGCCTAAAGATGGATCTGCCGCAGAATAAGTAGTTTGCGGGGTAAATGGCCACCCAGCCAGTAATCCGTACCGGTTGTGAAAACCAGAACCCCGAGATGGAAACTGAAACAAAGGTTCAAGGCCTACCGGGCACAACAAGCGCCAAAACTCCGCCATGCGAGCCATCGCGACGGGGGAAAACCAAGTACCACTCCTAACGGGGTGGTTTTTCCGAAGTGGAAAAAGCCTCCAGGAATAAGAACCTGGGCCAGAACCGTGGCCAGCCGCCGCCGTTACACCCGCCAGCTCGAGTTGTTGGCCGGTTTTATTGGGGCCTAAAGCCGGTCCGTAGCCCGTTTTGATAAGGTCTCTCTGGTGAAATTCTACAGCTTAACCTGTGGGAATTGCTGGAGGATACTATTCAAGCTTGAAGCCGGGAGAAGCCTGGAAGTACTCCCGGGGGTAAGGGGTGAAATTCTATTATCCCCGGAAGACCAACTGGTGCCGAAGCGGTCCAGCCTGGAACCGAACTTGACCGTGAGTTACGAAAAGCCAAGGGGCGCGGACCGGAATAAAATAACCAGGGTAGTCCTGGCCGTAAACGATGTGAACTTGGTGGTGGGAATGGCTTCGAACTGCCCAATTGCCGAAAGGAAGCTGTAAATTCACCCGCCTTGGAAGTACGGTCGCAAGACTGGAACCTAAAAGGAATTGGCGGGGGGACACCACAACGCGTGGAGCCTGGCGGTTTTATTGGGATTCCACGCAGACATCTCACTCAGGGGCGACAGCAGAAATGATGGGCAGGTTGATGACCTTGCTTGACAAGCTGAAAAGGAGGTGCAT
>EF503697
TAAAATGACTAGCCTGCGAGTCACGCCGTAAGGCGTGGCATACAGGCTCAGTAACACGTAGTCAACATGCCCAAAGGACGTGGATAACCTCGGGAAACTGAGGATAAACCGCGATAGGCCAAGGTTTCTGGAATGAGCTATGGCCGAAATCTATATGGCCTTTGGATTGGACTGCGGCCGATCAGGCTGTTGGTGAGGTAATGGCCCACCAAACCTGTAACCGGTACGGGCTTTGAGAGAAGTAGCCCGGAGATGGGCACTGAGACAAGGGCCCAGGCCCTATGGGGCGCAGCAGGCGCGAAACCTCTGCAATAGGCGAAAGCCTGACAGGGTTACTCTGAGTGATGCCCGCTAAGGGTATCTTTTGGCACCTCTAAAAATGGTGCAGAATAAGGGGTGGGCAAGTCTGGTGTCAGCCGCCGCGGTAATACCAGCACCCCGAGTTGTCGGGACGATTATTGGGCCTAAAGCATCCGTAGCCTGTTCTGCAAGTCCTCCGTTAAATCCACCTGCTCAACGGATGGGCTGCGGAGGATACCGCAGAGCTAGGAGGCGGGAGAGGCAAACGGTACTCAGTGGGTAGGGGTAAAATCCATTGATCTACTGAAGACCACCAGTGGCGAAGGCGGTTTGCCAGAACGCGCTCGACGGTGAGGGATGAAAGCTGGGGGAGCAAACCGGATTAGATACCCGGGGTAGTCCCAGCTGTAAACGGATGCAGACTCGGGTGATGGGGTTGGCTTCCGGCCCAACCCCAATTGCCCCCAGGCGAAGCCCGTTAAGATCTTGCCGCCCTGTCAGATGTCAGGGCCGCCAATACTCGAAACCTTAAAAGGAAATTGGGCGCGGGAAAAGTCACCAAAAGGGGGTTGAAACCCTGCGGGTTATATATTGTAAACC
"""

rdp_expected_out = {\
 'AY800210 description field':('Root,Archaea,Euryarchaeota',0.9),\
 'EU883771': ('Root,Archaea,Euryarchaeota,Methanobacteria,Methanobacteriales,Methanobacteriaceae,Methanosphaera',0.92),\
 'EF503699':('Root,Archaea,Crenarchaeota,Thermoprotei',0.82),
 'random_seq':('Root',1.0),
 'DQ260310':('Root,Archaea,Euryarchaeota,Methanobacteria,Methanobacteriales,Methanobacteriaceae',0.93),
 'EF503697':('Root,Archaea,Crenarchaeota,Thermoprotei',0.88)}

if __name__ == '__main__':
    main()
