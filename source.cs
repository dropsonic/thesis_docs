/****************************************
* Options.cs
****************************************/
using CommandLine;
using CommandLine.Text;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Thesis.App
{
    enum Filter
    {
        Gaussian, Difference, Threshold
    }

    enum Normalization
    {
        Minimax, Standard
    }

    enum Metric
    {
        Euclid, SqrEuclid
    }

    enum ClusterDistanceType
    {
        KMeans, Nearest
    }

    class Options
    {
        [ValueOption(0)]
        public string FieldsDescription { get; set; }

        [OptionArray('a', "anregimes", HelpText = "Files with anomalie regimes samples.")]
        public string[] Samples { get; set; }

        [OptionArray('r', "nomregimes", HelpText="Files with nominal regimes samples (will be cleaned from anomalies).", Required = true)]
        public string[] NominalSamples { get; set; }

        [Option('f', "filter", DefaultValue = Filter.Gaussian,
            HelpText = "Anomalies filter type and value. Appropriate values: gaussian, difference (maxdiff-value), threshold (threshold-value).")]
        public Filter Filter { get; set; }

        [Option('n', "normalization", DefaultValue = Normalization.Minimax,
            HelpText = "Normalization type. Appropriate values: minimax, standard.")]
        public Normalization Normalization { get; set; }

        [Option('m', "metric", DefaultValue = Metric.Euclid,
            HelpText = "Distance metric. Appropriate values: euclid, sqreuclid.")]
        public Metric Metric { get; set; }

        [Option('d', "cdist", DefaultValue = ClusterDistanceType.Nearest,
            HelpText = "Cluster distance function. Appropriate values: kmeans, nearest.")]
        public ClusterDistanceType ClusterDistanceType { get; set; }

        [Option('v', "novalue", DefaultValue = "?",
            HelpText = "Non-existing values replacement string.")]
        public string NoValueReplacement { get; set; }

        [Option('o', "output")]
        public string OutputFile { get; set; }

        [ValueList(typeof(List<string>))]
        public List<string> Args { get; set; }

        [HelpOption]
        public string GetUsage()
        {
            var help = new HelpText
            {
                Heading = new HeadingInfo("Thesis: Intergrated System Health Management based on Data Mining techniques."),
                Copyright = new CopyrightInfo("Vladimir Panchenko, 03-617, Moscow Aviation Institute", 2013),
                AdditionalNewLineAfterOption = true,
                AddDashesToOption = true
            };
            help.AddPreOptionsLine("Usage: thesis.app fields.txt -r nominal1.txt nominal2.txt [-a regime1.txt regimeN.txt] [-f threshold 0.5] [-d kmeans] [-m sqreuclid] [-n standard] [-v N/A]");
            help.AddPostOptionsLine("Normalization stats are calculated based on first nominal regime sample.\n");
            help.AddOptions(this);
            return help;
        }
    }

    class FilterOptions
    {
        //[Option('v', "fvalue", HelpText = "Maximum difference for difference filter | Threshold value for threshold filter")]
        [ValueOption(0)]
        public double? Value { get; set; }
    }
}

/****************************************
* Program.cs
****************************************/
using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Thesis.DataCleansing;
using Thesis.DDMS;
using Thesis.Orca;

namespace Thesis.App
{
    class Program
    {
        static string _fieldsFile;
        static IAnomaliesFilter _filter;
        static IRecordParser<string> _parser;
        static IScaling _scaling;
        static SystemModel _model;
        static DistanceMetric _metric;
        static ClusterDistanceMetric _clusterDistMetric;
        static IList<Field> _fields;

        static bool _writeToFile;

        static string GetBinName(string name)
        {
            return String.Concat(name, ".bin");
        }

        static void ShowUsage()
        {
            Console.WriteLine(new Options().GetUsage());
            Console.WriteLine();
            Environment.Exit(-1);
        }

        static void AddNominalSample(string filename)
        {
            if (!File.Exists(filename))
                throw new ApplicationException("File not found.");

            using (IDataReader reader = new PlainTextReader(filename, _fieldsFile, _parser))    // read input data   
            using (IDataReader binReader = new BinaryDataReader(reader, GetBinName(filename)))  // convert input data to binary format
            using (IDataReader scaleReader = new ScaleDataReader(binReader, _scaling))       // scale input data
            {
                if (_fields == null)
                    _fields = reader.Fields;

                string shuffleFile = String.Concat(filename, ".shuffle");
                scaleReader.Shuffle(shuffleFile);
                IEnumerable<Outlier> outliers = Enumerable.Empty<Outlier>();

                string regimeName = Path.GetFileNameWithoutExtension(filename);

                using (IDataReader cases = new BinaryDataReader(shuffleFile))
                using (IDataReader references = new BinaryDataReader(shuffleFile))
                {
                    var orca = new OrcaAD(DistanceMetrics.Euclid, neighborsCount: 10);
                    outliers = orca.Run(cases, references, true);
                }

                File.Delete(shuffleFile);

                var anomalies = _filter.Filter(outliers);

                Console.WriteLine("\n%%%%%% {0} %%%%%%", regimeName);
                Console.WriteLine("Anomalies:");
                foreach (var anomaly in anomalies)
                    Console.WriteLine("  Id = {0}, Score = {1}", anomaly.Id, anomaly.Score);

                Console.WriteLine("%%%%%%%%%%%%%%%\n");

                using (IDataReader cleanReader = new CleanDataReader(scaleReader, anomalies)) // clean input data from anomalies
                {
                    _model.AddRegime(regimeName, cleanReader);
                }
            }
        }

        static void AddSample(string filename)
        {
            if (!File.Exists(filename))
                throw new ApplicationException("File not found.");

            string regimeName = Path.GetFileNameWithoutExtension(filename);

            using (IDataReader tempReader = new PlainTextReader(filename, _fieldsFile, _parser))
            using (IDataReader tempBinReader = new BinaryDataReader(tempReader, GetBinName(filename)))
            {
                using (IDataReader tempScaleReader = new ScaleDataReader(tempBinReader, _scaling))
                {
                    _model.AddRegime(regimeName, tempScaleReader);
                }
            }
        }

        static void Main(string[] args)
        {
            Contract.ContractFailed += (s, e) =>
            {
                Console.WriteLine("Something went wrong. Please contact the developer.");
            };

            var argsParser = CommandLine.Parser.Default;
            Options options = new Options();
            if (argsParser.ParseArgumentsStrict(args, options))
            {
                _writeToFile = !String.IsNullOrEmpty(options.OutputFile);

                // Filter type
                if (options.Filter == Filter.Gaussian)
                    _filter = new GaussianFilter();
                else
                {
                    FilterOptions filterOpts = new FilterOptions();
                    if (argsParser.ParseArgumentsStrict(options.Args.ToArray(), filterOpts))
                    {
                        if (filterOpts.Value == null)
                            ShowUsage();
                        switch (options.Filter)
                        {
                            case Filter.Difference:
                                _filter = new DifferenceFilter((double)filterOpts.Value);
                                break;
                            case Filter.Threshold:
                                _filter = new ThresholdFilter((double)filterOpts.Value);
                                break;
                            default:
                                _filter = new GaussianFilter();
                                break;
                        }
                    }
                }

                try
                {
                    _parser = new PlainTextParser(options.NoValueReplacement);
                    _fieldsFile = options.FieldsDescription;

                    IDataReader mainReader = new PlainTextReader(options.NominalSamples[0], options.FieldsDescription, _parser);
                    switch (options.Normalization)
                    {
                        case Normalization.Standard:
                            _scaling = new StandardScaling(mainReader);
                            break;
                        default:
                            _scaling = new MinmaxScaling(mainReader);
                            break;
                    }
                    mainReader.Dispose();

                    switch (options.Metric)
                    {
                        case Metric.SqrEuclid:
                            _metric = DistanceMetrics.SqrEuсlid;
                            break;
                        default:
                            _metric = DistanceMetrics.Euclid;
                            break;
                    }

                    switch(options.ClusterDistanceType)
                    {
                        case ClusterDistanceType.KMeans:
                            _clusterDistMetric = ClusterDistances.CenterDistance;
                            break;
                        default:
                            _clusterDistMetric = ClusterDistances.NearestBoundDistance;
                            break;
                    }

                    Console.WriteLine("Enter epsilon:");
                    double eps;
                    while (!double.TryParse(Console.ReadLine(), out eps))
                        Console.WriteLine("Wrong format. Please enter epsilon again.");

                    _model = new SystemModel(eps);

                    foreach (var nominalSample in options.NominalSamples)
                        AddNominalSample(nominalSample);

                    foreach (var sample in options.Samples)
                        AddSample(sample);


                    Console.WriteLine("Knowledge base has been created. {0} regime(s) total:", _model.Regimes.Count);
                    foreach (var regime in _model.Regimes)
                    {
                        Console.WriteLine("\n***** {0} *****", regime.Name);
                        Console.WriteLine("{0} cluster(s) in regime.", regime.Clusters.Count);
                        int i = 0;
                        foreach (var cluster in regime.Clusters)
                        {
                            Console.SetBufferSize(Console.BufferWidth, Console.BufferHeight + 10);
                            Console.WriteLine("  --------------------------");
                            Console.WriteLine("  Cluster #{0}:", ++i);
                            Console.WriteLine("  Lower bound: {0}", String.Join(" | ", _scaling.Unscale(cluster.LowerBound)));
                            Console.WriteLine("  Upper bound: {0}", String.Join(" | ", _scaling.Unscale(cluster.UpperBound)));
                            Console.WriteLine("  Appropriate discrete values: {0}", String.Join(" | ", cluster.DiscreteValues.Select(f => String.Join("; ", f))));
                        }
                        Console.WriteLine("  --------------------------");
                        Console.WriteLine("******************", regime.Name);
                    }

                    Console.WriteLine("\nEnter record, or press enter to quit.");

                    string line = String.Empty;
                    do
                    {
                        line = Console.ReadLine();
                        if (String.IsNullOrEmpty(line)) break;

                        var record = _parser.TryParse(line, _fields);
                        if (record == null)
                        {
                            Console.WriteLine("Wrong record format. Please enter record again.");
                            continue;
                        }

                        _scaling.Scale(record);
                        double distance;
                        Regime closest;
                        Regime currentRegime = _model.DetectRegime(record, out distance, out closest);
                        if (currentRegime == null)
                            Console.WriteLine("Anomaly behavior detected (closest regime: {0}, distance: {1:0.00000}).\n",
                                closest.Name, distance);
                        else
                            Console.WriteLine("Current regime: {0}\n", currentRegime.Name);
                    } while (true);
                }
                catch (DataFormatException dfex)
                {
                    Console.WriteLine("Wrong data format. {0}", dfex.Message);
                    Console.ReadLine();
                }
                catch (Exception ex)
                {
                    Console.WriteLine("Error: {0} Please contact the developer.", ex.Message);
                    Console.ReadLine();
                }
            }
        }
    }
}

/****************************************
* CleanDataReader.cs
****************************************/
using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Thesis.DataCleansing
{
    /// <summary>
    /// Decorator for IDataReader: filters all specified anomalies.
    /// </summary>
    public class CleanDataReader : IDataReader
    {
        private IDataReader _baseReader;
        private HashSet<int> _anomalies;

        public CleanDataReader(IDataReader baseReader, IEnumerable<Outlier> anomalies)
        {
            Contract.Requires<ArgumentNullException>(baseReader != null);
            Contract.Requires<ArgumentNullException>(anomalies != null);

            _baseReader = baseReader;
            _anomalies = new HashSet<int>(anomalies.Select(a => a.Id));
        }

        public IList<Field> Fields
        {
            get { return _baseReader.Fields; }
        }

        public Record ReadRecord()
        {
            var record = _baseReader.ReadRecord();
            if (EndOfData)
                return null;

            if (_anomalies.Contains(record.Id)) // if this record is anomaly
                return ReadRecord();            // go to next record
            else
                return record;
        }

        public void Reset()
        {
            _baseReader.Reset();
        }

        public bool EndOfData
        {
            get { return _baseReader.EndOfData; }
        }

        public int Index
        {
            get { return _baseReader.Index; }
        }

        public IEnumerator<Record> GetEnumerator()
        {
            Reset();
            var record = ReadRecord();
            while (record != null)
            {
                yield return record;
                record = ReadRecord();
            }
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return this.GetEnumerator();
        }

        #region IDisposable
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        private bool m_Disposed = false;

        protected virtual void Dispose(bool disposing)
        {
            if (!m_Disposed)
            {
                if (disposing)
                {
                    // Managed resources are released here.
                    _baseReader.Dispose();
                }

                // Unmanaged resources are released here.
                m_Disposed = true;
            }
        }

        ~CleanDataReader()
        {
            Dispose(false);
        }
        #endregion
    }
}

/****************************************
* DifferenceFilter.cs
****************************************/
using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Thesis.DataCleansing
{
    public class DifferenceFilter : IAnomaliesFilter
    {
        private double _delta;

        public DifferenceFilter(double delta)
        {
            Contract.Requires<ArgumentOutOfRangeException>(delta >= 0);

            _delta = delta;
        }

        public IEnumerable<Outlier> Filter(IEnumerable<Outlier> outliers)
        {
            Contract.Requires<ArgumentNullException>(outliers != null);

            var o = outliers.ToArray();
            if (o.Length < 2)
                return Enumerable.Empty<Outlier>();

            for (int i = o.Length - 1; i > 0; i--)
            {
                double diff = o[i-1].Score - o[i].Score;
                if (diff > _delta)
                    return o.Take(i);
            }

            return Enumerable.Empty<Outlier>();
        }
    }
}

/****************************************
* GaussianFilter.cs
****************************************/
using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Thesis.Orca;

namespace Thesis.DataCleansing
{
    /// <summary>
    /// Filters anomalies which score is greater then M+3σ.
    /// </summary>
    public class GaussianFilter : IAnomaliesFilter
    {
        public IEnumerable<Outlier> Filter(IEnumerable<Outlier> outliers)
        {
            Contract.Requires<ArgumentNullException>(outliers != null);

            var anomalies = new List<Outlier>();
            double cutoff = GetCutoff(outliers);
            foreach (var outlier in outliers)
            {
                if (outlier.Score > cutoff)
                    anomalies.Add(outlier);
            }
            return anomalies;
        }

        private IDictionary<double, double> GetFrequencies(IList<Outlier> outliers)
        {
            return outliers.GroupBy(o => o.Score)
                           .ToDictionary(gr => gr.Key,
                                         gr => (double)gr.Count() / (double)outliers.Count);
        }

        private double GetCutoff(IEnumerable<Outlier> outliers)
        {
            var freq = GetFrequencies(outliers.ToList());
            double mean, disp;
            GetStatValues(freq, out mean, out disp);
            double cutoff = mean + 3 * Math.Sqrt(disp);
            return cutoff;
        }

        /// <summary>
        /// Calculates mean value and dispersion by one pass through frequencies dictionary.
        /// </summary>
        private void GetStatValues(IDictionary<double, double> freq, out double mean, out double dispersion)
        {
            mean = 0;
            dispersion = 0;
            foreach (var f in freq)
            {
                double x = f.Key * f.Value;
                mean += x;
                dispersion += x * f.Key;
            }

            dispersion -= mean * mean;
        }

        // Slow
        //private double Mean(IDictionary<double, double> freq)
        //{
        //    return freq.Select(v => v.Key * v.Value).Sum();
        //}

        //private double Dispersion(IDictionary<double, double> freq, double mean)
        //{
        //    return freq.Select(v => v.Key * v.Key * v.Value).Sum()
        //           - Math.Pow(Mean(freq), 2);
        //}
    }
}

/****************************************
* IAnomaliesFilter.cs
****************************************/
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Thesis.Orca;

namespace Thesis.DataCleansing
{
    public interface IAnomaliesFilter
    {
        /// <summary>
        /// Return anomalies in all outliers.
        /// </summary>
        IEnumerable<Outlier> Filter(IEnumerable<Outlier> outliers);
    }
}

/****************************************
* ThresholdFilter.cs
****************************************/
using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Thesis.DataCleansing
{
    /// <summary>
    /// Filters anomalies which score (reduced by min score) is greater then threshold.
    /// </summary>
    public class ThresholdFilter : IAnomaliesFilter
    {
        private double _threshold;

        public ThresholdFilter(double threshold)
        {
            Contract.Requires<ArgumentOutOfRangeException>(threshold >= 0);

            _threshold = threshold;
        }

        public IEnumerable<Outlier> Filter(IEnumerable<Outlier> outliers)
        {
            double min = outliers.Min().Score;
            foreach (var outlier in outliers)
            {
                if (outlier.Score - min > _threshold)
                    yield return outlier;
            }
        }
    }
}

/****************************************
* Cluster.cs
****************************************/
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Thesis
{
    public class Cluster
    {
        public float[] UpperBound { get; set; }
        public float[] LowerBound { get; set; }

        /// <summary>
        /// Appropriate discrete values for this cluster.
        /// </summary>
        public HashSet<int>[] DiscreteValues { get; set; }

        internal Cluster(Record init)
        {
            LowerBound = (float[])init.Real.Clone();
            UpperBound = (float[])init.Real.Clone();
            DiscreteValues = init.Discrete.Select(i =>
                {
                    var hs = new HashSet<int>();
                    hs.Add(i);
                    return hs;
                }).ToArray();
        }

        /// <summary>
        /// Determines if record lies inside cluster boundaries.
        /// </summary>
        internal bool Contains(Record record)
        {
            int realFieldsCount = UpperBound.Length;
            for (int i = 0; i < realFieldsCount; i++)
            {
                if (record.Real[i] > UpperBound[i])
                    return false;
                if (record.Real[i] < LowerBound[i])
                    return false;
            }

            int discreteFieldsCount = DiscreteValues.Length;
            for (int i = 0; i < discreteFieldsCount; i++)
            {
                if (!DiscreteValues[i].Contains(record.Discrete[i]))
                    return false;
            }

            return true;
        }

        /// <summary>
        /// Adds record to the cluster (expands cluster bounds, if necessary).
        /// </summary>
        internal void Add(Record record)
        {
            // Expand cluster bounds
            int realFieldsCount = UpperBound.Length;
            for (int i = 0; i < realFieldsCount; i++)
            {
                if (record.Real[i] > UpperBound[i])
                    UpperBound[i] = record.Real[i];
                else if (record.Real[i] < LowerBound[i])
                    LowerBound[i] = record.Real[i];
            }

            // Add discrete values to cluster appropriate values
            int discreteFieldsCount = DiscreteValues.Length;
            for (int i = 0; i < discreteFieldsCount; i++)
                DiscreteValues[i].Add(record.Discrete[i]);
        }
    }
}

/****************************************
* ClusterDatabase.cs
****************************************/
using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Thesis.DDMS
{
    class ClusterDatabase
    {
        private List<Cluster> _clusters = new List<Cluster>();
        private double _eps;
        private Weights _weights;
        private ClusterDistanceMetric _distance;
        private DistanceMetric _metric;

        private bool IsEmpty
        {
            get { return _clusters.Count == 0; }
        }

        public int Size
        {
            get { return _clusters.Count; }
        }

        public IReadOnlyCollection<Cluster> Clusters
        {
            get { return _clusters.AsReadOnly(); }
        }

        public ClusterDatabase(double eps, Weights weights, ClusterDistanceMetric distanceFunc, DistanceMetric metric)
        {
            Contract.Requires<ArgumentOutOfRangeException>(eps >= 0);
            Contract.Requires<ArgumentNullException>(weights != null);
            Contract.Requires<ArgumentNullException>(distanceFunc != null);
            Contract.Requires<ArgumentNullException>(metric != null);

            _eps = eps;
            _weights = weights;
            _distance = distanceFunc;
            _metric = metric;
        }

        public void AddRecord(Record record)
        {
            if (IsEmpty) // if cluster database empty
            {
                // form input vector into cluster and insert into cluster database
                AddCluster(new Cluster(record));
            }
            else
            {
                // if record is inside at least one cluster, do nothing;
                // else:
                if (!_clusters.Any(c => c.Contains(record)))
                {
                    double dist;
                    Cluster closest = FindClosest(record, out dist);
                    // if distance to the closest cluster is greater then epsilon
                    if (dist > _eps)
                        // add new cluster initialized by this record
                        AddCluster(new Cluster(record));
                    else
                        // add record to the closest cluster
                        closest.Add(record);
                }
            }
        }

        /// <summary>
        /// Determines if record is inside or close enough
        /// to at least one cluster in database.
        /// </summary>
        public bool Contains(Record record)
        {
            // if distance to the closest cluster is less then epsilon, return true
            return Distance(record) <= _eps;
        }

        /// <summary>
        /// Calculates distance from record to the closest cluster. 
        /// </summary>
        public double Distance(Record record)
        {
            if (record == null)
                return double.PositiveInfinity;

            // if record is inside at least one cluster
            if (_clusters.Any(c => c.Contains(record)))
                return 0;

            double dist;
            Cluster closest = FindClosest(record, out dist);
            return dist;
        }

        private void AddCluster(Cluster cluster)
        {
            if (cluster != null)
                _clusters.Add(cluster);
        }

        /// <summary>
        /// Finds closest cluster from specified record.
        /// </summary>
        private Cluster FindClosest(Record record, out double dist)
        {
            dist = double.PositiveInfinity;
            Cluster closest = null;
            foreach (var cluster in _clusters)
            {
                double d = _distance(cluster, record, _weights, _metric);
                if (d < dist)
                {
                    dist = d;
                    closest = cluster;
                }
            }

            return closest;
        }
    }
}

/****************************************
* ClusterDistanceMetric.cs
****************************************/
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Thesis.DDMS
{
    public delegate double ClusterDistanceMetric(Cluster c, Record x, Weights weights, DistanceMetric metric);
}

/****************************************
* ClusterDistances.cs
****************************************/
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Thesis.DDMS
{
    public static class ClusterDistances
    {
        /// <summary>
        /// Distance from record to the center of cluster.
        /// </summary>
        public static double CenterDistance(Cluster cluster, Record record, Weights weights, DistanceMetric metric)
        {
            int realFieldsCount = record.Real.Length;
            int discreteFieldsCount = record.Discrete.Length;

            // Calculate center of the cluster
            Record center = new Record();
            center.Real = new float[realFieldsCount];
            center.Discrete = new int[discreteFieldsCount];
            for (int i = 0; i < realFieldsCount; i++)
                center.Real[i] = (cluster.LowerBound[i] + cluster.UpperBound[i]) / 2;
            for (int i = 0; i < discreteFieldsCount; i++)
                // if cluster contains value, keep it the same as in record
                center.Discrete[i] = cluster.DiscreteValues[i].Contains(record.Discrete[i]) ? record.Discrete[i] : -1;

            return metric(center, record, weights);
        }

        /// <summary>
        /// Distance from record to the nearest cluster boundary line.
        /// </summary>
        public static double NearestBoundDistance(Cluster cluster, Record record, Weights weights, DistanceMetric metric)
        {
            int realFieldsCount = record.Real.Length;
            int discreteFieldsCount = record.Discrete.Length;

            // Calculate nearest boundary of the cluster
            Record nearestBound = new Record();
            nearestBound.Real = new float[realFieldsCount];
            nearestBound.Discrete = new int[discreteFieldsCount];
            for (int i = 0; i < realFieldsCount; i++)
            {
                if (record.Real[i] >= cluster.LowerBound[i])
                {
                    if (record.Real[i] <= cluster.UpperBound[i])
                        nearestBound.Real[i] = record.Real[i];
                    else
                        nearestBound.Real[i] = cluster.UpperBound[i];
                }
                else
                {
                    nearestBound.Real[i] = cluster.LowerBound[i];
                }
            }
            for (int i = 0; i < discreteFieldsCount; i++)
                // if cluster contains value, keep it the same as in record
                nearestBound.Discrete[i] = cluster.DiscreteValues[i].Contains(record.Discrete[i]) ? record.Discrete[i] : -1;

            return metric(nearestBound, record, weights);
        }
    }
}

/****************************************
* Regime.cs
****************************************/
using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Thesis.DDMS
{
    /// <summary>
    /// Regime of the system.
    /// </summary>
    public class Regime
    {
        private string _name;
        private ClusterDatabase _db;

        public IReadOnlyCollection<Cluster> Clusters
        {
            get { return _db.Clusters; }
        }

        public string Name
        {
            get { return _name; }
        }

        public Regime(string name, double eps, IDataReader records, ClusterDistanceMetric distanceFunc, DistanceMetric metric)
        {
            Contract.Requires<ArgumentOutOfRangeException>(eps >= 0);
            Contract.Requires<ArgumentNullException>(records != null);
            Contract.Requires<ArgumentNullException>(distanceFunc != null);
            Contract.Requires<ArgumentNullException>(metric != null);

            _name = name;
            _db = new ClusterDatabase(eps, records.Fields.Weights(), distanceFunc, metric);
            foreach (var record in records)
                _db.AddRecord(record);
        }

        /// <summary>
        /// Determines if record is inside or close enough
        /// to at least one cluster in the regime.
        /// </summary>
        public bool Contains(Record record)
        {
            return _db.Contains(record);
        }

        /// <summary>
        /// Calculates distance from record to the regime. 
        /// </summary>
        public double Distance(Record record)
        {
            return _db.Distance(record);
        }
    }
}

/****************************************
* SystemModel.cs
****************************************/
using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Thesis.DDMS
{
    /// <summary>
    /// Represents all regimes of work for the system.
    /// </summary>
    public class SystemModel
    {
        private ClusterDistanceMetric _distanceFunc;
        private DistanceMetric _metric;
        private List<Regime> _regimes = new List<Regime>();
        private double _eps;

        public IReadOnlyCollection<Regime> Regimes
        {
            get { return _regimes.AsReadOnly(); }
        }

        public SystemModel(double eps)
            : this(eps, ClusterDistances.NearestBoundDistance, DistanceMetrics.Euclid)
        { }

        public SystemModel(double eps, ClusterDistanceMetric distanceFunc, DistanceMetric metric)
        {
            Contract.Requires<ArgumentOutOfRangeException>(eps >= 0);
            Contract.Requires<ArgumentNullException>(distanceFunc != null);
            Contract.Requires<ArgumentNullException>(metric != null);

            _distanceFunc = distanceFunc;
            _metric = metric;
            _eps = eps;
        }

        public void AddRegime(string name, IDataReader records)
        {
            Contract.Requires<ArgumentNullException>(records != null);

            Regime regime = new Regime(name, _eps, records, _distanceFunc, _metric);
            _regimes.Add(regime);
        }

        /// <summary>
        /// Detects closest regime to the record.
        /// Returns null, if record is further then epsilon from all regimes.
        /// </summary>
        public Regime DetectRegime(Record record)
        {
            double distance;
            Regime closest;
            return DetectRegime(record, out distance, out closest);
        }

        /// <summary>
        /// Detects closest regime to the record and calculates distance between them.
        /// Returns null, if record is further then epsilon from all regimes.
        /// </summary>
        public Regime DetectRegime(Record record, out double distance, out Regime closestRegime)
        {
            double mind = double.PositiveInfinity;
            closestRegime = null;

            foreach (var regime in _regimes)
            {
                double d = regime.Distance(record);
                if (d < mind)
                {
                    mind = d;
                    closestRegime = regime;
                }
            }

            distance = mind;
            return mind > _eps ? null : closestRegime;
        }
    }
}

/****************************************
* BinaryShuffle.cs
****************************************/
using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Thesis
{
    class BinaryShuffle
    {
        private IDataReader _sourceReader;
        private IDataReader _shuffleReader;

        private string _shuffleFile;

        private int _iterations;
        private int _randFilesCount;
        private Random _rand;

        /// <param name="sourceReader">Reader for input data.</param>
        /// <param name="shuffleFile">Name of shuffle binary file.</param>
        /// <param name="iterations">Number of shuffle iterations.</param>
        /// <param name="randFilesCount">Number of random part-files.</param>
        public BinaryShuffle(IDataReader sourceReader, string shuffleFile, 
                             int iterations = 5, int randFilesCount = 10)
        {
            Contract.Requires<ArgumentNullException>(sourceReader != null);
            Contract.Requires<ArgumentException>(!String.IsNullOrEmpty(shuffleFile));
            Contract.Requires<ArgumentOutOfRangeException>(iterations >= 1);
            Contract.Requires<ArgumentOutOfRangeException>(randFilesCount >= 1);

            _sourceReader = sourceReader;
            _shuffleFile = shuffleFile;
            _iterations = iterations;
            _randFilesCount = randFilesCount;

            _rand = new Random((int)DateTime.Now.Ticks);

            MultiShuffle();
        }


        private void SetFileReader(string filename)
        {
            Contract.Requires(!String.IsNullOrEmpty(filename));

            // if infile_ already points to a file, close it
            if (_shuffleReader != null)
                _shuffleReader.Dispose();

            _shuffleReader = new BinaryDataReader(filename);
        }

        private void ResetFileReader()
        {
            if (_shuffleReader != null)
                _shuffleReader.Dispose();
        }

        public void MultiShuffle()
        {
            Shuffle(_sourceReader, _shuffleFile);
            _shuffleReader = new BinaryDataReader(_shuffleFile);
            
            for (int i = 1; i < _iterations; i++)
            {
                Shuffle(_shuffleReader, _shuffleFile);
                SetFileReader(_shuffleFile);
            }

            ResetFileReader();
        }

        private void Shuffle(IDataReader sourceReader, string filename)
        {
            //-------------------------
            // set up tmp file names
            //
            string[] tmpFileNames = new string[_randFilesCount];
            for (int i = 0; i < _randFilesCount; i++)
                tmpFileNames[i] = filename + ".tmp." + i.ToString();


            //-------------------------------
            // open files for writing
            //
            IDataWriter[] tmpFilesOut = new BinaryDataWriter[_randFilesCount];
            try
            {
                for (int i = 0; i < tmpFileNames.Length; i++)
                    tmpFilesOut[i] = new BinaryDataWriter(tmpFileNames[i], _sourceReader.Fields);

                //--------------------------------
                // read in data file and randomly shuffle examples to
                // temporary files
                //
                foreach (var rec in _sourceReader)
                {
                    int index = _rand.Next(tmpFilesOut.Length);
                    tmpFilesOut[index].WriteRecord(rec);
                }
            }
            finally
            {
                // close temporary files
                for (int i = 0; i < tmpFilesOut.Length; i++)
                    if (tmpFilesOut[i] != null)
                        tmpFilesOut[i].Dispose();
            }

            //-------------------------------
            // open tmpfiles for reading 
            //

            IDataReader[] tmpFilesIn = new BinaryDataReader[_randFilesCount];
            try
            {
                for (int i = 0; i < tmpFilesIn.Length; i++)
                    tmpFilesIn[i] = new BinaryDataReader(tmpFileNames[i]);

                //-----------------------------------
                // open final destination file
                //

                ResetFileReader(); // closes original file

                using (IDataWriter outfile = new BinaryDataWriter(filename, _sourceReader.Fields))
                {
                    //--------------------------------------
                    // concatenate tmp files in random order
                    //
                    int[] order = new int[_randFilesCount];
                    for (int i = 0; i < _randFilesCount; i++)
                        order[i] = i;

                    // The modern version of the Fisher–Yates shuffle (the Knuth shuffle)
                    for (int i = order.Length - 1; i >= 0; i--)
                    {
                        int j = _rand.Next(i + 1);
                        int temp = order[i];
                        order[i] = order[j];
                        order[j] = temp;
                    }

                    for (int i = 0; i < order.Length; i++)
                    {
                        IDataReader infile = tmpFilesIn[order[i]];
                        foreach (var rec in infile)
                            outfile.WriteRecord(rec);
                    }
                }
            }
            finally
            {
                // close temporary files
                for (int i = 0; i < tmpFilesIn.Length; i++)
                    if (tmpFilesIn[i] != null)
                        tmpFilesIn[i].Dispose();
            }

            //-------------------------------
            // delete tmpfiles 
            //
            foreach (var fileName in tmpFileNames)
                File.Delete(fileName);

            ResetFileReader();
        }
    }
}

/****************************************
* DistanceMetric.cs
****************************************/
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Thesis
{
    public delegate double DistanceMetric(Record a, Record b, Weights weights);
}

/****************************************
* DistanceMetrics.cs
****************************************/
using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Thesis
{
    public static class DistanceMetrics
    {
        /// <summary>
        /// Squared Euclid distance.
        /// </summary>
        public static double SqrEuсlid(Record a, Record b, Weights weights)
        {
            Contract.Requires(a.Real.Length == b.Real.Length);
            Contract.Requires(a.Discrete.Length == b.Discrete.Length);
            Contract.Requires(a.Real.Length == weights.Real.Length);
            Contract.Requires(a.Discrete.Length == weights.Discrete.Length);

            int realFieldsCount = a.Real.Length;
            int discreteFieldsCount = a.Discrete.Length;

            double d = 0;

            // real 
            for (int i = 0; i < realFieldsCount; i++)
            {
                // check for missing values
                int missingCount = 0;
                if (float.IsNaN(a.Real[i]))
                    missingCount++;
                if (float.IsNaN(b.Real[i]))
                    missingCount++;

                if (missingCount == 0)
                {
                    double diff = a.Real[i] - b.Real[i];
                    d += diff * diff * weights.Real[i];
                }
                // one value is missing
                else if (missingCount == 1)
                {
                    d += weights.Real[i];
                }
            }

            // discrete
            for (int i = 0; i < discreteFieldsCount; i++)
            {
                if (a.Discrete[i] != b.Discrete[i])
                    d += weights.Discrete[i];
            }

            return d;
        }

        public static double Euclid(Record a, Record b, Weights weights)
        {
            return Math.Sqrt(SqrEuсlid(a, b, weights));
        }
    }
}

/****************************************
* Field.cs
****************************************/
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Diagnostics.Contracts;

namespace Thesis
{
    public class Field
    {
        public enum FieldType
        {
            Discrete = 0,
            /// <summary>
            /// Discrete Data Driven
            /// </summary>
            DiscreteDataDriven = 1,
            Continuous = 2,
            IgnoreFeature = 3
        }

        public string Name { get; set; }

        public FieldType Type { get; set; }

        /// <summary>
        /// List of values for discrete field.
        /// </summary>
        public IList<string> Values { get; set; }

        private float _weight = float.NaN;
        public float Weight
        {
            get { return _weight; }
            set { _weight = value; }
        }

        public bool HasWeight 
        { 
            get { return !float.IsNaN(Weight); } 
        }
    }
}

/****************************************
* FieldsHelper.cs
****************************************/
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Thesis
{
    public static class FieldsHelper
    {
        public static int RealCount(this IEnumerable<Field> fields)
        {
            return fields.Count(f => f.Type == Field.FieldType.Continuous);
        }

        public static int DiscreteCount(this IEnumerable<Field> fields)
        {
            return fields.Count(f => f.Type == Field.FieldType.Discrete ||
                                     f.Type == Field.FieldType.DiscreteDataDriven);
        }

        public static Weights Weights(this IEnumerable<Field> fields)
        {
            var real = fields.Where(f => f.Type == Field.FieldType.Continuous).Select(f => f.Weight).ToArray();
            var discrete = fields.Where(f => f.Type == Field.FieldType.Discrete)
                                     .Concat(fields.Where(f => f.Type == Field.FieldType.DiscreteDataDriven))
                                     .Select(f => f.Weight).ToArray();
            return new Weights() { Real = real, Discrete = discrete };
        }
    }
}

/****************************************
* Outlier.cs
****************************************/
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Thesis
{
    [DebuggerDisplay("Id = {Id}; Score = {Score}")]
    public struct Outlier : IComparable<Outlier>
    {
        //public Record Record { get; set; }
        public int Id { get; set; }
        public double Score { get; set; }
        //public IEnumerable<int> Neighbors { get; set; }

        public int CompareTo(Outlier other)
        {
            return Score.CompareTo(other.Score);
        }
    }
}

/****************************************
* Record.cs
****************************************/
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Thesis
{
    [DebuggerDisplay("Id = {Id}")]
    public class Record : ICloneable
    {
        public int Id { get; set; }
        public float[] Real { get; set; }
        public int[] Discrete { get; set; }

        public Record() { }

        public Record(int id, float[] real, int[] discrete)
        {
            Id = id;
            Real = real;
            Discrete = discrete;
        }

        object ICloneable.Clone()
        {
            return this.Clone();
        }

        public Record Clone()
        {
            return new Record(Id, (float[])Real.Clone(), (int[])Discrete.Clone());
        }
    }
}

/****************************************
* Weights.cs
****************************************/
using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Thesis
{
    public class Weights
    {
        public float[] Real { get; set; }
        public float[] Discrete { get; set; }

        public static Weights Identity(int realFieldsCount, int discreteFieldsCount)
        {
            Contract.Requires<ArgumentOutOfRangeException>(realFieldsCount >= 0);
            Contract.Requires<ArgumentOutOfRangeException>(discreteFieldsCount >= 0);

            float[] real = Enumerable.Repeat(1f, realFieldsCount).ToArray();
            float[] discrete = Enumerable.Repeat(1f, discreteFieldsCount).ToArray();
            
            return new Weights()
            {
                Real = real,
                Discrete = discrete
            };
        }
    }
}

/****************************************
* BinaryHeap.cs
****************************************/
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Thesis.Collections
{
    public class BinaryHeap<T> : ICollection, IEnumerable<T> where T : IComparable
    {
        private const int _defaultCapacity = 4;

        private List<T> _items;

        public BinaryHeap()
            : this(_defaultCapacity) { }

        public BinaryHeap(int capacity)
        {
            if (capacity < 0)
                throw new ArgumentOutOfRangeException("capacity");
            _items = new List<T>(capacity);
        }

        public BinaryHeap(IEnumerable<T> collection)
        {
            _items = new List<T>(collection);
            BuildHeap(_items);
        }

        private static void SwapElements(IList<T> list, int a, int b)
        {
            T temp = list[a];
            list[a] = list[b];
            list[b] = temp;
        }

        private static void BuildHeap(IList<T> a)
        {
            for (int i = a.Count / 2; i > 0; i--)
                Heapify(a, i);
        }

        private static void Heapify(IList<T> a, int i)
        {
            int left = 2 * i;
            int right = 2 * i + 1;
            int largest = i;

            if (left <= a.Count && a[left - 1].CompareTo(a[i - 1]) > 0)
                largest = left;
            if (right <= a.Count && a[right - 1].CompareTo(a[largest - 1]) > 0)
                largest = right;

            if (largest != i)
            {
                SwapElements(a, i - 1, largest - 1);
                Heapify(a, largest);
            }
        }

        private static void HeapIncreaseKey(IList<T> a, int i, T key)
        {
            a[i - 1] = key;

            for (int j = i; j > 1 && a[j / 2 - 1].CompareTo(a[j - 1]) < 0; j = j / 2)
                SwapElements(a, j - 1, j / 2 - 1);

            //for (int j = i; j > 0 && a[j].CompareTo(a[j - 1]) > 0; j--)
            //    SwapElements(a, j, j - 1);
        }

        public int IndexOf(T item)
        {
            return _items.IndexOf(item);
        }

        public T this[int index]
        {
            get
            {
                return _items[index];
            }
            set
            {
                HeapIncreaseKey(_items, index + 1, value);
            }
        }

        public void Push(T item)
        {
            _items.Add(item);
            HeapIncreaseKey(_items, _items.Count, item);
        }

        public void Clear()
        {
            _items.Clear();
        }

        public bool Contains(T item)
        {
            return _items.Contains(item);
        }

        public void CopyTo(T[] array, int arrayIndex)
        {
            _items.CopyTo(array, arrayIndex);
        }

        public IEnumerator<T> GetEnumerator()
        {
            return _items.GetEnumerator();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return _items.GetEnumerator();
        }

        /// <summary>
        /// Removes and returns max element of the heap.
        /// </summary>
        /// <returns>Max element of the heap</returns>
        public T Pop()
        {
            if (Count == 0)
                throw new InvalidOperationException("Heap is empty.");
            
            T max = this[0];
            if (Count == 1)
            {
                _items.Remove(max);
                return max;
            }

            T last = _items[_items.Count - 1];
            _items.Remove(last);
            _items[0] = last;
            Heapify(_items, 1);
            return max;
        }

        public T Peek()
        {
            if (Count == 0)
                throw new InvalidOperationException("Heap is empty.");
            return this[0];
        }

        #region ICollection
        public int Count
        {
            get { return _items.Count; }
        }

        public void CopyTo(Array array, int index)
        {
            ((ICollection)_items).CopyTo(array, index);
        }

        public bool IsSynchronized
        {
            get { return false; }
        }

        public object SyncRoot
        {
            get { return this; }
        }
        #endregion
    }
}

/****************************************
* BatchDataReader.cs
****************************************/
using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Thesis
{
    public class BatchDataReader : IDataReader
    {
        IDataReader _baseReader;

        int _batchSize;
        int _lastOffset;

        public int Offset { get; private set; }

        public IList<Record> CurrentBatch { get; private set; }

        public BatchDataReader(IDataReader baseReader, int batchSize)
        {
            Contract.Requires<ArgumentNullException>(baseReader != null);
            Contract.Requires<ArgumentOutOfRangeException>(batchSize > 0);

            _baseReader = baseReader;
            _baseReader.Reset();
            _batchSize = batchSize;
        }

        /// <summary>
        /// Reads next batch of records.
        /// </summary>s
        public bool GetNextBatch()
        {
            CurrentBatch = new List<Record>(_batchSize);

            int nr = 0;
            for (int i = 0; i < _batchSize; i++)
            {
                if (_baseReader.EndOfData)
                    break;
                CurrentBatch.Add(_baseReader.ReadRecord());
                nr++;
            }

            Offset += _lastOffset;
            _lastOffset = nr;

            return nr > 0;
        }

        #region IDataReader
        public IList<Field> Fields
        {
            get { return _baseReader.Fields; }
        }

        public Record ReadRecord()
        {
            return _baseReader.ReadRecord();
        }

        public void Reset()
        {
            _baseReader.Reset();
            Offset = 0;
            _lastOffset = 0;
            CurrentBatch = null;
        }

        public bool EndOfData
        {
            get { return _baseReader.EndOfData; }
        }

        public int Index
        {
            get { return _baseReader.Index; }
        }

        public IEnumerator<Record> GetEnumerator()
        {
            Reset();
            var record = ReadRecord();
            while (record != null)
            {
                yield return record;
                record = ReadRecord();
            }
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return this.GetEnumerator();
        }
        #endregion

        #region IDisposable
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        private bool m_Disposed = false;

        protected virtual void Dispose(bool disposing)
        {
            if (!m_Disposed)
            {
                if (disposing)
                {
                    // Managed resources are released here.
                    _baseReader.Dispose();
                }

                // Unmanaged resources are released here.
                m_Disposed = true;
            }
        }

        ~BatchDataReader()
        {
            Dispose(false);
        }
        #endregion
    }
}

/****************************************
* DataFormatException.cs
****************************************/
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Thesis
{
    public class DataFormatException : FormatException
    {
        public DataFormatException()
            : base() { }

        public DataFormatException(string message)
            : base(message) { }

        public DataFormatException(string message, Exception innerException)
            : base(message, innerException) { }
    }
}

/****************************************
* DataHelper.cs
****************************************/
using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Thesis
{
    public static class DataHelper
    {
        public static void CopyTo(this IDataReader reader, IDataWriter destination)
        {
            foreach (var record in reader)
                destination.WriteRecord(record);
        }

        public static void Shuffle(this IDataReader reader, string outputFile,
                                   int iterations = 5, int randFilesCount = 10)
        {
            Contract.Requires<ArgumentException>(!String.IsNullOrEmpty(outputFile));
            Contract.Requires<ArgumentOutOfRangeException>(iterations >= 1);
            Contract.Requires<ArgumentOutOfRangeException>(randFilesCount >= 1);

            BinaryShuffle shuffle = new BinaryShuffle(reader, outputFile, iterations, randFilesCount);
            shuffle.MultiShuffle();
        }
    }
}

/****************************************
* IDataReader.cs
****************************************/
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Thesis
{
    public interface IDataReader: IEnumerable<Record>, IDisposable
    {
        IList<Field> Fields { get; }

        Record ReadRecord();

        void Reset();

        bool EndOfData { get; }

        int Index { get; }
    }
}

/****************************************
* IDataWriter.cs
****************************************/
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Thesis
{
    public interface IDataWriter : IDisposable
    {
        void WriteRecord(Record record);
        int Count { get; }
    }
}

/****************************************
* IRecordParser.cs
****************************************/
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Thesis
{
    public interface IRecordParser<T>
    {
        Record Parse(T input, IList<Field> fields);

        /// <summary>
        /// Returns null if parsing failed.
        /// </summary>
        Record TryParse(T input, IList<Field> fields);
    }
}

/****************************************
* ScaleDataReader.cs
****************************************/
using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Thesis
{
    /// <summary>
    /// Scales every record on reading with defined scaling method.
    /// </summary>
    public class ScaleDataReader : IDataReader
    {
        private IDataReader _baseReader;
        private IScaling _scaling;

        public ScaleDataReader(IDataReader baseReader, IScaling scaling)
        {
            Contract.Requires<ArgumentNullException>(baseReader != null);
            Contract.Requires<ArgumentNullException>(scaling != null);

            _baseReader = baseReader;
            _scaling = scaling;
        }

        #region IDataReader
        public IList<Field> Fields
        {
            get { return _baseReader.Fields; }
        }

        public Record ReadRecord()
        {
            var record = _baseReader.ReadRecord();
            if (record != null)
                _scaling.Scale(record);
            return record;
        }

        public void Reset()
        {
            _baseReader.Reset();
        }

        public bool EndOfData
        {
            get { return _baseReader.EndOfData; }
        }

        public int Index
        {
            get { return _baseReader.Index; }
        }

        public IEnumerator<Record> GetEnumerator()
        {
            Reset();
            var record = ReadRecord();
            while (record != null)
            {
                yield return record;
                record = ReadRecord();
            }
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return this.GetEnumerator();
        }
        #endregion

        #region IDisposable
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        private bool m_Disposed = false;

        protected virtual void Dispose(bool disposing)
        {
            if (!m_Disposed)
            {
                if (disposing)
                {
                    // Managed resources are released here.
                    _baseReader.Dispose();
                }

                // Unmanaged resources are released here.
                m_Disposed = true;
            }
        }

        ~ScaleDataReader()
        {
            Dispose(false);
        }
        #endregion
    }
}

/****************************************
* BinaryDataReader.cs
****************************************/
using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Thesis
{
    /// <summary>
    /// Represents binary record format file reader.
    /// </summary>
    public class BinaryDataReader : IDataReader
    {
        BinaryReader _infile;

        public int RecordsCount { get; private set; }
        public int RealFieldsCount { get; private set; }
        public int DiscreteFieldsCount { get; private set; }

        private long _dataOffset; // 0 + size of header

        private IList<Field> _fields;

        private readonly string _tempFile;

        public BinaryDataReader(string filename)
        {
            InitReader(filename);
        }

        /// <summary>
        /// Copies all record from source to binary file using BinaryDataWriter
        /// and opens data reader on that file.
        /// </summary>
        /// <param name="source">Source data reader.</param>
        /// <param name="tempFile">Temporary binary file name.</param>
        /// <param name="keepBinary">true, if binary file should be deleted hereafter.</param>
        public BinaryDataReader(IDataReader source, string tempFile, bool keepBinary = false)
        {
            var writer = new BinaryDataWriter(source, tempFile);
            writer.Dispose();
            if (!keepBinary)
                _tempFile = tempFile;
            InitReader(tempFile);
        }

        /// <summary>
        /// Initializes reader.
        /// </summary>
        /// <param name="filename">Input data file name.</param>
        private void InitReader(string filename)
        {
            _infile = new BinaryReader(File.OpenRead(filename));
            _fields = new List<Field>();
            ReadHeader();
            SeekPosition(0);
        }

        private void ReadHeader()
        {
            long oldPos = _infile.BaseStream.Position;

            _infile.BaseStream.Position = 0;
            RecordsCount = _infile.ReadInt32();
            RealFieldsCount = _infile.ReadInt32();
            DiscreteFieldsCount = _infile.ReadInt32();
            
            int fieldsCount = _infile.ReadInt32();
            for (int i = 0; i < fieldsCount; i++)
                _fields.Add(ReadField());

            _dataOffset = _infile.BaseStream.Position;
            _infile.BaseStream.Position = oldPos;
        }

        private Field ReadField()
        {
            string name = _infile.ReadString();
            Field.FieldType type = (Field.FieldType)_infile.ReadInt32();
            float weight = _infile.ReadSingle();

            bool hasValues = _infile.ReadBoolean();
            List<string> values = null;
            if (hasValues)
            {
                int valuesCount = _infile.ReadInt32();
                values = new List<string>();
                for (int i = 0; i < valuesCount; i++)
                    values.Add(_infile.ReadString());
            }

            return new Field() { Name = name, Type = type, Weight = weight, Values = values };
        }

        private void SeekPosition(int pos)
        {
            Contract.Requires<ArgumentOutOfRangeException>(pos >= 0);
            Contract.Requires<ArgumentOutOfRangeException>(pos <= RecordsCount);

            long filepos = _dataOffset + pos * 
                (sizeof(int) + RealFieldsCount * sizeof(float) + DiscreteFieldsCount * sizeof(int));
            
            _infile.BaseStream.Position = filepos;
            Index = pos;
        }

        #region IDataReader
        public int Index { get; set; }

        public IList<Field> Fields
        {
            get { return _fields; }
        }

        public Record ReadRecord()
        {
            if (EndOfData)
                return null;

            var id = _infile.ReadInt32();
            var real = _infile.ReadFloatArray(RealFieldsCount);
            var discrete = _infile.ReadIntArray(DiscreteFieldsCount);

            Index++;

            return new Record(id, real, discrete);
        }

        public void Reset()
        {
            SeekPosition(0);
        }

        public bool EndOfData
        {
            get { return Index == RecordsCount; }
        }

        public IEnumerator<Record> GetEnumerator()
        {
            Reset();
            while (!EndOfData)
                yield return ReadRecord();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return this.GetEnumerator();
        }
        #endregion

        #region IDisposable
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        private bool m_Disposed = false;

        protected virtual void Dispose(bool disposing)
        {
            if (!m_Disposed)
            {
                if (disposing)
                {
                    // Managed resources are released here.
                    _infile.Close();
                    if (!String.IsNullOrEmpty(_tempFile))
                        File.Delete(_tempFile);
                }

                // Unmanaged resources are released here.
                m_Disposed = true;
            }
        }

        ~BinaryDataReader()
        {
            Dispose(false);
        }
        #endregion
    }
}

/****************************************
* BinaryDataWriter.cs
****************************************/
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using System.Diagnostics.Contracts;

namespace Thesis
{
    /// <summary>
    /// Represents Orca format binary file writer.
    /// </summary>
    public class BinaryDataWriter : IDataWriter
    {
        BinaryWriter _outfile;

        bool _headerWritten = false;

        IList<Field> _fields;

        private int _realFieldsCount;
        private int _discreteFieldsCount;

        private int _count = 0; // number of records


        public BinaryDataWriter(string filename, IList<Field> fields)
        {
            Contract.Requires<ArgumentException>(!String.IsNullOrEmpty(filename));
            Contract.Requires<ArgumentNullException>(fields != null);

            _outfile = new BinaryWriter(File.Create(filename));
            
            _fields = fields;
            _realFieldsCount = fields.RealCount();
            _discreteFieldsCount = fields.DiscreteCount();

            WriteHeader();
        }


        /// <summary>
        /// Creates new BinaryDataWriter and copies all records from IDataReader source.
        /// </summary>
        public BinaryDataWriter(IDataReader source, string filename)
            : this(filename, source.Fields)
        {
            foreach (var record in source)
                WriteRecord(record);

            WriteHeader(Count);
        }

        private void WriteHeader()
        {
            //long oldPos = _outfile.BaseStream.Position;

            _outfile.Seek(0, SeekOrigin.Begin);
            _outfile.Write(_count); // number of records
            _outfile.Write(_realFieldsCount);
            _outfile.Write(_discreteFieldsCount);

            _outfile.Write(_fields.Count);
            foreach (var field in _fields)
                WriteField(field);

            //_outfile.BaseStream.Position = oldPos;
            _headerWritten = true;
        }

        private void WriteHeader(int numRecords)
        {
            if (!_headerWritten)
                WriteHeader();

            long oldPos = _outfile.BaseStream.Position;
            _outfile.Seek(0, SeekOrigin.Begin);
            _outfile.Write(numRecords);
            _outfile.BaseStream.Position = oldPos;
        }

        private void WriteField(Field field)
        {
            _outfile.Write(field.Name);
            _outfile.Write((int)field.Type);
            _outfile.Write(field.Weight);

            bool hasValues = field.Values != null;
            _outfile.Write(hasValues);
            if (hasValues)
            {
                _outfile.Write(field.Values.Count);
                foreach (var value in field.Values)
                    _outfile.Write(value);
            }
        }

        public void WriteRecord(Record record)
        {
            if (record == null)
                return;

            if (record.Real.Length != _realFieldsCount ||
                record.Discrete.Length != _discreteFieldsCount)
                throw new ArgumentException("Wrong number of values in record.");

            if (!_headerWritten)
                WriteHeader();

            _outfile.Write(record.Id);
            if (_realFieldsCount > 0)
                _outfile.Write(record.Real);
            if (_discreteFieldsCount > 0)
                _outfile.Write(record.Discrete);

            _count++;
        }

        public int Count
        {
            get { return _count; }
        }

        #region IDisposable
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }
 
        private bool m_Disposed = false;
 
        protected virtual void Dispose(bool disposing)
        {
            if (!m_Disposed)
            {
                if (disposing)
                {
                // Managed resources are released here.
                    WriteHeader(_count);
                    _outfile.Close();
                }
 
                // Unmanaged resources are released here.
                m_Disposed = true;
            }
        }
 
        ~BinaryDataWriter()    
        {        
            Dispose(false);
        }
        #endregion
    }
}

/****************************************
* BinaryHelper.cs
****************************************/
using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Thesis
{
    internal static class BinaryHelper
    {
        public static void Write(this BinaryWriter writer, float[] data)
        {
            Contract.Requires<ArgumentNullException>(data != null);

            byte[] binaryData = new byte[data.Length * sizeof(float)];
            System.Buffer.BlockCopy(data, 0, binaryData, 0, binaryData.Length);
            writer.Write(binaryData);
        }

        public static void Write(this BinaryWriter writer, int[] data)
        {
            Contract.Requires<ArgumentNullException>(data != null);

            byte[] binaryData = new byte[data.Length * sizeof(int)];
            System.Buffer.BlockCopy(data, 0, binaryData, 0, binaryData.Length);
            writer.Write(binaryData);
        }

        public static float[] ReadFloatArray(this BinaryReader reader, int count)
        {
            Contract.Requires<ArgumentOutOfRangeException>(count >= 0);

            if (count == 0)
                return new float[0];

            float[] result = new float[count];
            byte[] binaryData = reader.ReadBytes(sizeof(float) * count);
            Buffer.BlockCopy(binaryData, 0, result, 0, binaryData.Length);

            return result;
        }

        public static int[] ReadIntArray(this BinaryReader reader, int count)
        {
            Contract.Requires<ArgumentOutOfRangeException>(count >= 0);

            if (count == 0)
                return new int[0];

            int[] result = new int[count];
            byte[] binaryData = reader.ReadBytes(sizeof(int) * count);
            Buffer.BlockCopy(binaryData, 0, result, 0, binaryData.Length);

            return result;
        }
    }
}

/****************************************
* PlainTextParser.cs
****************************************/
using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Globalization;

namespace Thesis
{
    public class PlainTextParser : IRecordParser<string>
    {
        private char[] _recordDelimiters;
        private string _noValueReplacement;

        public PlainTextParser(char[] recordDelimiters,
                               string noValueReplacement = "?")
        {
            Contract.Requires<ArgumentNullException>(recordDelimiters != null);
            Contract.Requires<ArgumentException>(recordDelimiters.Length > 0);
            Contract.Requires<ArgumentException>(!String.IsNullOrEmpty(noValueReplacement));

            _recordDelimiters = recordDelimiters;
            _noValueReplacement = noValueReplacement;
        }

        public PlainTextParser(string noValueReplacement = "?")
            : this(new char[] { ',', ';' }, noValueReplacement)
        { }

        /// <summary>
        /// Parses string and returns record. Returns null if string is a comment.
        /// </summary>
        /// <exception cref="Thesis.DataFormatException"/>
        public Record Parse(string input, IList<Field> fields)
        {
            Contract.Requires<ArgumentNullException>(fields != null);

            var tokens = StringHelper.Tokenize(input, _recordDelimiters);

            if (tokens.Length == 0) // if comment
                return null;

            // check to make sure there are the correct number of tokens
            if (tokens.Length != fields.Count)
                throw new DataFormatException("Wrong number of tokens.");

            int realFieldsCount = fields.Count(f => f.Type == Field.FieldType.Continuous);
            int discreteFieldsCount = fields.Count(f => f.Type == Field.FieldType.Discrete ||
                                                      f.Type == Field.FieldType.DiscreteDataDriven);

            var real = new float[realFieldsCount];
            var discrete = new int[discreteFieldsCount];
            int iReal = 0;
            int iDiscrete = 0;


            for (int i = 0; i < fields.Count; i++)
            {
                if (fields[i].Type == Field.FieldType.IgnoreFeature)
                    continue;
                if (tokens[i] == _noValueReplacement)
                {
                    switch (fields[i].Type)
                    {
                        case Field.FieldType.Continuous:
                            real[iReal++] = float.NaN; break;
                        case Field.FieldType.Discrete:
                        case Field.FieldType.DiscreteDataDriven:
                            discrete[iDiscrete++] = -1; break;
                    }
                }
                else
                {
                    switch (fields[i].Type)
                    {
                        case Field.FieldType.Continuous:
                            real[iReal++] = float.Parse(tokens[i], CultureInfo.InvariantCulture); break;
                        case Field.FieldType.Discrete:
                            int value = fields[i].Values.IndexOf(tokens[i]);
                            if (value != -1)
                                discrete[iDiscrete++] = value;
                            else
                                throw new DataFormatException(String.Format(
                                    "Discrete value '{0}' for field '{1}' doesn't exist.", tokens[i], fields[i]));
                            break;
                        case Field.FieldType.DiscreteDataDriven:
                            int valuec = fields[i].Values.IndexOf(tokens[i]);
                            if (valuec != -1)
                                discrete[iDiscrete++] = valuec;
                            else
                            {
                                // Add new value to the field description.
                                fields[i].Values.Add(tokens[i]);
                                discrete[iDiscrete++] = fields[i].Values.Count - 1;
                            }
                            break;
                    }
                }
            }

            return new Record(0, real, discrete);
        }

        public Record TryParse(string input, IList<Field> fields)
        {
            Contract.Requires<ArgumentNullException>(fields != null);

            try
            {
                return Parse(input, fields);
            }
            catch (DataFormatException)
            {
                return null;
            }
        }
    }
}

/****************************************
* PlainTextReader.cs
****************************************/
using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Thesis
{
    public class PlainTextReader : IDataReader
    {
        private IRecordParser<string> _parser;

        private StreamReader _infile;
        private List<Field> _fields = new List<Field>();

        private char[] _fieldsDelimiters;

        private float _realWeight;
        private float _discreteWeight;

        public int _realFieldsCount;
        public int _discreteFieldsCount;

        public PlainTextReader(string dataFile, string fieldsFile, IRecordParser<string> parser,
                               float realWeight = 1.0f, float discreteWeight = 0.4f)
            : this(dataFile, fieldsFile, parser,
                   new char[] { ',', ':', ';' }, realWeight, discreteWeight)
        { }

        public PlainTextReader(string dataFile, string fieldsFile, IRecordParser<string> parser, 
                               char[] fieldsDelimiters,
                               float realWeight = 1.0f, float discreteWeight = 0.4f)
        {
            Contract.Requires<ArgumentException>(!String.IsNullOrEmpty(dataFile));
            Contract.Requires<ArgumentException>(!String.IsNullOrEmpty(fieldsFile));
            Contract.Requires<ArgumentNullException>(parser != null);
            Contract.Requires<ArgumentNullException>(fieldsDelimiters != null);
            Contract.Requires<ArgumentException>(fieldsDelimiters.Length > 0);

            _parser = parser;

            _realWeight = realWeight;
            _discreteWeight = discreteWeight;
            _fieldsDelimiters = fieldsDelimiters;

            LoadFields(fieldsFile);
            _infile = new StreamReader(dataFile);
            Index = 0;
        }


        private void LoadFields(string filename)
        {
            Contract.Requires(!String.IsNullOrEmpty(filename));

            using (var infile = new StreamReader(filename))
            {
                while (!infile.EndOfStream)
                {
                    string line = infile.ReadLine();
                    var tokens = StringHelper.Tokenize(line, _fieldsDelimiters);
                    if (tokens.Length > 0)
                    {
                        Field newField = CreateField(tokens);
                        _fields.Add(newField);
                    }
                }
            }

            _realFieldsCount = _fields.Count(f => f.Type == Field.FieldType.Continuous);
            _discreteFieldsCount = _fields.Count(f => f.Type == Field.FieldType.Discrete ||
                                                      f.Type == Field.FieldType.DiscreteDataDriven);
        }

        private Field CreateField(string[] tokens)
        {
            Contract.Requires<ArgumentNullException>(tokens != null);
            Contract.Requires<ArgumentException>(tokens.Length > 0);

            Field field = new Field();

            field.Weight = float.NaN; // no weight

            int i = 0; // start token
            float weight;
            if (float.TryParse(tokens[0], out weight)) // if weight is defined
            {
                field.Weight = weight;
                i++;
            }

            field.Name = tokens[i++];
            string sType = tokens[i];

            if (tokens.Length == i)
                field.Type = Field.FieldType.IgnoreFeature;
            else
            {
                switch (sType)
                {
                    case "ignore":
                        field.Type = Field.FieldType.IgnoreFeature;
                        break;
                    case "continuous":
                        field.Type = Field.FieldType.Continuous;
                        if (!field.HasWeight)
                            field.Weight = _realWeight;
                        break;
                    case "discrete":
                        field.Type = Field.FieldType.DiscreteDataDriven;
                        field.Values = new List<string>();
                        if (!field.HasWeight)
                            field.Weight = _discreteWeight;
                        break;
                    default:
                        //Discrete type of field: adding all of it's values
                        field.Type = Field.FieldType.Discrete;
                        field.Values = new List<string>(tokens.Length - 1);
                        for (int j = 1; j < tokens.Length; j++)
                            field.Values.Add(tokens[j]);
                        if (!field.HasWeight)
                            field.Weight = _discreteWeight;
                        break;
                }
            }

            return field;
        }

        #region IDataReader
        public IList<Field> Fields
        {
            get { return _fields.AsReadOnly(); }
        }

        public Record ReadRecord()
        {
            if (!EndOfData)
            {
                string line = _infile.ReadLine();
                var record = _parser.Parse(line, _fields);
                if (record == null) // if comment
                    return ReadRecord(); // go to next line
                else
                {
                    Index++;
                    record.Id = Index;
                    return record;
                }
            }
            else
            {
                return null;
            }
        }

        public void Reset()
        {
            Index = 0;
            _infile.BaseStream.Position = 0;
        }

        public bool EndOfData
        {
            get { return _infile.EndOfStream; }
        }

        public int Index { get; private set; }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return this.GetEnumerator();
        }

        public IEnumerator<Record> GetEnumerator()
        {
            Reset();
            var record = ReadRecord();
            while (record != null)
            {
                yield return record;
                record = ReadRecord();
            }
        }
        #endregion

        #region IDisposable
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }
 
        private bool m_Disposed = false;
 
        protected virtual void Dispose(bool disposing)
        {
            if (!m_Disposed)
            {
                if (disposing)
                {
                // Managed resources are released here.
                    _infile.Close();
                }
 
                // Unmanaged resources are released here.
                m_Disposed = true;
            }
        }
 
        ~PlainTextReader()    
        {        
            Dispose(false);
        }
        #endregion
    }
}

/****************************************
* StringHelper.cs
****************************************/
using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Thesis
{
    static class StringHelper
    {
        public static string[] Tokenize(string line, char[] delimiters)
        {
            Contract.Requires<ArgumentNullException>(line != null);
            Contract.Requires<ArgumentNullException>(delimiters != null);
            Contract.Requires<ArgumentException>(delimiters.Length > 0, "No delimiters specified.");

            // Strip comments (original; remove comments in this version)
            int index = line.IndexOf('%');
            if (index >= 0)
                line = line.Remove(index);
            // Replace tab characters
            line = line.Replace('\t', ' ');
            // Split string into tokens
            var tokens = line.Split(delimiters, StringSplitOptions.RemoveEmptyEntries);
            // Trim whitespaces in tokens
            tokens = tokens.Select(s => s.Trim()).ToArray();

            return tokens;
        }
    }
}

/****************************************
* IScaling.cs
****************************************/
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Thesis
{
    public interface IScaling
    {
        void Scale(Record record);
        void Unscale(Record record);

        float[] Scale(float[] real);
        float[] Unscale(float[] real);
    }
}

/****************************************
* MinmaxScaling.cs
****************************************/
using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Thesis
{
    /// <summary>
    /// Scales values by min and max value (from zero to one).
    /// </summary>
    public class MinmaxScaling : IScaling
    {
        private int _realFieldsCount;

        private float[] _min;
        private float[] _range;

        private IDataReader _data;

        public MinmaxScaling(IDataReader data)
        {
            _data = data;
            _realFieldsCount = data.Fields.RealCount();

            GetDataProperties();
        }

        /// <summary>
        /// Calculates data properties (min and range).
        /// </summary>
        private void GetDataProperties()
        {
            // initialize vectors 
            _min = new float[_realFieldsCount];
            float[] max = new float[_realFieldsCount];
            _range = new float[_realFieldsCount];
            for (int i = 0; i < _realFieldsCount; i++)
            {
                _min[i] = float.MaxValue;
                max[i] = float.MinValue;
            }

            foreach (var record in _data)
            {
                for (int i = 0; i < _realFieldsCount; i++)
                {
                    float value = record.Real[i];
                    if (!float.IsNaN(value))
                    {
                        if (value < _min[i])
                            _min[i] = value;
                        else if (value > max[i])
                            max[i] = value;
                    }
                }
            }

            // calculate range
            for (int i = 0; i < _realFieldsCount; i++)
                _range[i] = max[i] - _min[i];

            _data.Reset();
        }

        public void Scale(Record record)
        {
            for (int i = 0; i < _realFieldsCount; i++)
                record.Real[i] = _range[i] == 0 ? 0 : (record.Real[i] - _min[i]) / _range[i];
        }

        public void Unscale(Record record)
        {
            for (int i = 0; i < _realFieldsCount; i++)
                record.Real[i] = record.Real[i] * _range[i] + _min[i];
        }

        public float[] Scale(float[] real)
        {
            Contract.Requires<ArgumentNullException>(real != null);
            Contract.Assert(real.Length == _realFieldsCount);

            float[] result = new float[_realFieldsCount];
            for (int i = 0; i < _realFieldsCount; i++)
                result[i] = _range[i] == 0 ? 0 : (real[i] - _min[i]) / _range[i];

            return result;
        }

        public float[] Unscale(float[] real)
        {
            Contract.Requires<ArgumentNullException>(real != null);
            Contract.Assert(real.Length == _realFieldsCount);

            float[] result = new float[_realFieldsCount];
            for (int i = 0; i < _realFieldsCount; i++)
                result[i] = real[i] * _range[i] + _min[i];

            return result;
        }
    }
}

/****************************************
* StandardScaling.cs
****************************************/
using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Thesis
{
    /// <summary>
    /// Standardize real values (scales to mean and standard deviations).
    /// </summary>
    public class StandardScaling : IScaling
    {
        private int _realFieldsCount;

        private float[] _mean;
        private float[] _std;

        private IDataReader _data;

        public StandardScaling(IDataReader data)
        {
            _data = data;
            _realFieldsCount = data.Fields.RealCount();

            GetDataProperties();
        }

        /// <summary>
        /// Calculates data properties (min and range).
        /// </summary>
        private void GetDataProperties()
        {
            // initialize vectors 
            _mean = new float[_realFieldsCount];
            _std = new float[_realFieldsCount];
            double[] sumv = new double[_realFieldsCount];
            double[] sumsqv = new double[_realFieldsCount];
            int[] num = new int[_realFieldsCount];

            foreach (var record in _data)
            {
                for (int i = 0; i < _realFieldsCount; i++)
                {
                    if (!float.IsNaN(record.Real[i]))
                    {
                        double r = ((double)record.Real[i]);
                        sumv[i] += r;
                        sumsqv[i] += r * r;
                        num[i]++;
                    }
                }

                for (int i = 0; i < _realFieldsCount; i++)
                {
                    if (num[i] > 1)
                    {
                        double meanValue = sumv[i] / num[i];
                        _mean[i] = double.IsNaN(meanValue) ? 0 : ((float)meanValue);

                        double stdValue = Math.Sqrt((sumsqv[i] - sumv[i] * sumv[i] / num[i]) / (num[i] - 1));
                        _std[i] = double.IsNaN(stdValue) ? 0 : ((float)stdValue);
                    }
                    else
                    {
                        _mean[i] = 0;
                        _std[i] = 0;
                    }
                }
            }

            _data.Reset();
        }

        public void Scale(Record record)
        {
            for (int i = 0; i < _realFieldsCount; i++)
                record.Real[i] = _std[i] == 0 ? 0 : (record.Real[i] - _mean[i]) / _std[i];
        }

        public void Unscale(Record record)
        {
            for (int i = 0; i < _realFieldsCount; i++)
                record.Real[i] = record.Real[i] * _std[i] + _mean[i];
        }

        public float[] Scale(float[] real)
        {
            Contract.Requires<ArgumentNullException>(real != null);
            Contract.Assert(real.Length == _realFieldsCount);

            float[] result = new float[_realFieldsCount];
            for (int i = 0; i < _realFieldsCount; i++)
                result[i] = _std[i] == 0 ? 0 : (real[i] - _mean[i]) / _std[i];

            return result;
        }

        public float[] Unscale(float[] real)
        {
            Contract.Requires<ArgumentNullException>(real != null);
            Contract.Assert(real.Length == _realFieldsCount);

            float[] result = new float[_realFieldsCount];
            for (int i = 0; i < _realFieldsCount; i++)
                result[i] = real[i] * _std[i] + _mean[i];

            return result;
        }
    }
}

/****************************************
* NeighborsDistance.cs
****************************************/
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Thesis.Collections;

namespace Thesis.Orca
{
    struct NeighborsDistance
    {
        public Record Record { get; set; }
        public BinaryHeap<double> Distances { get; set; }
    }
}

/****************************************
* OrcaAD.cs
****************************************/
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Diagnostics.Contracts;
using Thesis.Collections;
using System.Diagnostics;

namespace Thesis.Orca
{
    public class OrcaAD
    {
        private ScoreFunction _scoreFunction;
        private DistanceMetric _distanceFunction;
        private int _numOutliers;
        private int _neighborsCount;
        private double _cutoff;
        private int _batchSize;

        #region Constructors
        /// <param name="distanceFunction">Distance function for calculating the distance between two examples with weights.</param>
        /// <param name="numOutliers">Number of outliers.</param>
        /// <param name="neighborsCount"></param>
        /// <param name="cutoff"></param>
        /// <param name="batchSize"></param>
        public OrcaAD(DistanceMetric distanceFunction,
            int numOutliers = 30, int neighborsCount = 5,
            double cutoff = 0, int batchSize = 1000)
            : this(ScoreFunctions.Average, distanceFunction, numOutliers,
                neighborsCount, cutoff, batchSize)
        { }

        /// <param name="scoreFunction">Distance score function.</param>
        /// <param name="numOutliers">Number of outliers.</param>
        /// <param name="neighborsCount"></param>
        /// <param name="cutoff"></param>
        /// <param name="batchSize"></param>
        public OrcaAD(ScoreFunction scoreFunction,
            int numOutliers = 30, int neighborsCount = 5,
            double cutoff = 0, int batchSize = 1000)
            : this(scoreFunction, DistanceMetrics.SqrEuсlid, numOutliers,
                neighborsCount, cutoff, batchSize)
        { }

        /// <param name="numOutliers">Number of outliers.</param>
        /// <param name="neighborsCount"></param>
        /// <param name="cutoff"></param>
        /// <param name="batchSize"></param>
        public OrcaAD(int numOutliers = 30, int neighborsCount = 5,
            double cutoff = 0, int batchSize = 1000)
            : this(ScoreFunctions.Average, DistanceMetrics.SqrEuсlid, numOutliers,
                neighborsCount, cutoff, batchSize)
        { }

        /// <param name="scoreFunction">Distance score function.</param>
        /// <param name="distanceFunction">Distance function for calculating the distance between two examples with weights.</param>
        /// <param name="numOutliers">Number of outliers.</param>
        /// <param name="neighborsCount"></param>
        /// <param name="cutoff"></param>
        /// <param name="batchSize"></param>
        public OrcaAD(ScoreFunction scoreFunction,
            DistanceMetric distanceFunction,
            int numOutliers = 30, int neighborsCount = 5, 
            double cutoff = 0, int batchSize = 1000)
        {
            Contract.Requires<ArgumentNullException>(scoreFunction != null);
            Contract.Requires<ArgumentNullException>(distanceFunction != null);
            Contract.Requires<ArgumentOutOfRangeException>(numOutliers > 0);
            Contract.Requires<ArgumentOutOfRangeException>(neighborsCount > 0);
            Contract.Requires<ArgumentOutOfRangeException>(cutoff >= 0);
            Contract.Requires<ArgumentOutOfRangeException>(batchSize > 0);

            _scoreFunction = scoreFunction;
            _distanceFunction = distanceFunction;
            _numOutliers = numOutliers;
            _neighborsCount = neighborsCount;
            _cutoff = cutoff;
            _batchSize = batchSize;
        }
        #endregion

        /// <param name="cases">Data reader for input data.</param>
        /// <param name="references">Data reader for reference data (can't be the same reader object).</param>
        /// <param name="returnAll">If true, returns score info for all records in input data.</param>
        /// <returns></returns>
        public IEnumerable<Outlier> Run(IDataReader cases, IDataReader references, bool returnAll = false)
        {
            Contract.Requires<ArgumentNullException>(cases != null);
            Contract.Requires<ArgumentNullException>(references != null);
            Contract.Requires<ArgumentException>(!object.ReferenceEquals(cases, references));

            // Test cases
            using (BatchDataReader batchInFile = new BatchDataReader(cases, _batchSize))
            // Reference database
            {
                List<Outlier> outliers = new List<Outlier>();
                bool done = false;
                double cutoff = _cutoff;
                Weights weights = cases.Fields.Weights();

                //-----------------------
                // run the outlier search 
                //
                done = !batchInFile.GetNextBatch(); //start batch
                while (!done)
                {
                    Trace.PrintRecords(batchInFile.CurrentBatch);

                    var o = FindOutliers(batchInFile, references, weights, cutoff);
                    outliers.AddRange(o);

                    references.Reset();

                    //-------------------------------
                    // sort the current best outliers 
                    // and keep the best
                    //
                    outliers.Sort();
                    outliers.Reverse(); // sorting in descending order
                    int numOutliers = _numOutliers;
                    if (outliers.Count > numOutliers &&
                        outliers[numOutliers - 1].Score > cutoff)
                    {
                        // New cutoff
                        cutoff = outliers[numOutliers - 1].Score;
                    }
                    done = !batchInFile.GetNextBatch();
                }

                return returnAll ? outliers : outliers.Take(_numOutliers);
            }
        }

        private IList<Outlier> FindOutliers(BatchDataReader cases, IDataReader references, Weights weights, double cutoff)
        {
            Contract.Requires<ArgumentNullException>(cases != null);
            Contract.Requires<ArgumentNullException>(references != null);

            int k = _neighborsCount; // number of neighbors
            
            var records = new List<Record>(cases.CurrentBatch);
            int batchRecCount = records.Count;


            // distance to neighbors — Neighbors(b) in original description
            var neighborsDist = new List<NeighborsDistance>(batchRecCount);
            // initialize distance score with max distance
            for (int i = 0; i < batchRecCount; i++)
            {
                var kDistDim = new NeighborsDistance() 
                { 
                    Record = records[i],
                    Distances = new BinaryHeap<double>(k) 
                };
                for (int j = 0; j < k; j++)
                    kDistDim.Distances.Push(double.MaxValue);
                neighborsDist.Add(kDistDim);
            }

            // vector to store furthest nearest neighbour
            var minkDist = new List<double>(batchRecCount);
            for (int i = 0; i < neighborsDist.Count; i++)
                minkDist.Add(double.MaxValue);

            // candidates stores the integer index
            var candidates = Enumerable.Range(0, batchRecCount).ToList();

            int neighborsDist_i;
            int minkDist_i;
            int candidates_i;

            // loop over objects in reference table
            foreach (var descRecord in references)
            {
                neighborsDist_i = 0;
                minkDist_i = 0;
                candidates_i = 0;

                for (int j = 0; j < batchRecCount; j++)
                {
                    double dist = _distanceFunction(records[j], descRecord, weights);

                    if (dist < minkDist[minkDist_i])
                    {
                        if (cases.Offset + candidates[candidates_i] != references.Index - 1)
                        {
                            BinaryHeap<double> kvec = neighborsDist[neighborsDist_i].Distances;
                            kvec.Push(dist);
                            kvec.Pop();
                            minkDist[minkDist_i] = kvec.Peek();

                            double score = _scoreFunction(kvec);

                            if (score <= cutoff)
                            {
                                candidates.RemoveAt(candidates_i--);
                                records.RemoveAt(j--); batchRecCount--;
                                neighborsDist.RemoveAt(neighborsDist_i--);
                                minkDist.RemoveAt(minkDist_i--);

                                if (candidates.Count == 0)
                                    break;
                            }
                        }
                    }

                    neighborsDist_i++;
                    minkDist_i++;
                    candidates_i++;
                }

                if (candidates.Count == 0)
                    break;

                Trace.Message(String.Format("Offset: {0} | Ref #{1} processed.", cases.Offset, references.Index));
            }

            //--------------------------------
            // update the list of top outliers 
            // 
            candidates_i = 0;
            List<Outlier> outliers = new List<Outlier>();

            foreach (var point in neighborsDist)
            {
                Outlier outlier = new Outlier();
                //outlier.Record = point.Record;
                outlier.Id = point.Record.Id;
                outlier.Score = _scoreFunction(point.Distances);
                outliers.Add(outlier);
            }

            return outliers;
        }
    }
}

/****************************************
* ScoreFunction.cs
****************************************/
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Thesis.Orca
{
    public delegate double ScoreFunction(IEnumerable<double> distances);
}

/****************************************
* ScoreFunctions.cs
****************************************/
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Thesis.Orca
{
    public static class ScoreFunctions
    {
        private static readonly ScoreFunction _average = new ScoreFunction(dist => dist.Sum() / dist.Count());
        private static readonly ScoreFunction _sum = new ScoreFunction(dist => dist.Sum());
        private static readonly ScoreFunction _kthNeighbor = new ScoreFunction(dist => dist.FirstOrDefault());

        /// <summary>
        /// Average distance to k neighbors.
        /// </summary>
        public static ScoreFunction Average
        {
            get { return _average; }
        }

        /// <summary>
        /// Total distance to k neighbors (sum).
        /// </summary>
        public static ScoreFunction Sum
        {
            get { return _sum; }
        }

        /// <summary>
        /// Distance to kth neighbor.
        /// </summary>
        public static ScoreFunction KthNeighbor
        {
            get { return _kthNeighbor; }
        }
    }
}

/****************************************
* Trace.cs
****************************************/
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Thesis.Orca
{
    static class Trace
    {
        [Conditional("DEBUG")]
        public static void PrintRecords(IEnumerable<Record> records)
        {
            Console.WriteLine("-----------TRACE-----------");
            IEnumerable<Record> pRecords = records.Count() <= 10 ? records :
                records.Take(10);
            foreach (var record in pRecords)
            {
                Console.Write("#{0}: ", record.Id);
                foreach (var real in record.Real)
                    Console.Write("{0} ", real);
                Console.Write("| ");
                foreach (var discrete in record.Discrete)
                    Console.Write("{0} ", discrete);
                Console.WriteLine();
            }
            Console.WriteLine("-----------END TRACE-----------");
            Console.WriteLine();
        }

        [Conditional("DEBUG")]
        public static void Message(string message)
        {
            Console.WriteLine("TRACE: " + message);
        }
    }
}
