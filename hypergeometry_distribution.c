#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <complex.h>
#include <time.h>
#include <stdbool.h>
#include <float.h>

// =============================
// CONSTANTS & CONFIGURATION
// =============================
#define MAX_DIMENSIONS 256
#define HYPERCUBE_VERTICES 65536  // 2^16 for manageable computation
#define HYPERSPHERE_SAMPLES 10000
#define MONTE_CARLO_ITERATIONS 100000
#define EPSILON 1e-12

// Mathematical constants
#define PI 3.14159265358979323846
#define E 2.71828182845904523536
#define GOLDEN_RATIO 1.61803398874989484820

// =============================
// DATA STRUCTURES
// =============================
typedef struct {
    int dimensions;
    double coordinates[MAX_DIMENSIONS];
    double weight;
    double curvature;
    bool is_vertex;
} HyperPoint;

typedef struct {
    HyperPoint points[HYPERCUBE_VERTICES];
    int dimension;
    long double volume;
    long double surface_area;
    double eccentricity[MAX_DIMENSIONS];
} HyperGeometry;

typedef struct {
    double lambda;
    double kappa;
    double theta;
    double phi;
    double scale_factor;
} HypergeometricParameters;

typedef struct {
    double distance_distribution[1000];
    double angle_distribution[1000];
    double curvature_distribution[1000];
    int sample_count;
} HyperStatisticalData;

// =============================
// MATHEMATICAL CORE FUNCTIONS
// =============================

// Gamma function approximation using Lanczos method
double gamma_function(double x) {
    if (x <= 0) return INFINITY;
    if (x < 1e-10) return 1.0 / x;
    
    double p[] = {
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7
    };
    
    double result = 0.99999999999980993;
    double temp = x + 6.5;
    
    for (int i = 0; i < 7; i++) {
        result += p[i] / (x + i + 1);
    }
    
    result = sqrt(2 * PI) * pow(temp, x + 0.5) * exp(-temp) * result;
    
    return result;
}

// Beta function: B(x,y) = Γ(x)Γ(y)/Γ(x+y)
double beta_function(double x, double y) {
    if (x <= 0 || y <= 0) return INFINITY;
    return gamma_function(x) * gamma_function(y) / gamma_function(x + y);
}

// Hypergeometric function ₂F₁ (Gauss hypergeometric function)
double complex hypergeometric_2F1(double a, double b, double c, double complex z) {
    if (cabs(z) >= 1.0) {
        // Use analytic continuation or transformation
        z = 1.0 / (1.0 - z);
        a = c - a;
        b = c - b;
    }
    
    double complex result = 1.0;
    double complex term = 1.0;
    
    for (int n = 1; n < 1000; n++) {
        term *= (a + n - 1) * (b + n - 1) / (c + n - 1) * z / n;
        result += term;
        
        if (cabs(term) < EPSILON) break;
    }
    
    return result;
}

// Volume of n-dimensional hypersphere
double hypersphere_volume(int n, double radius) {
    if (n <= 0) return 0;
    
    double numerator = pow(PI, n/2.0) * pow(radius, n);
    double denominator = gamma_function(n/2.0 + 1);
    
    return numerator / denominator;
}

// Surface area of n-dimensional hypersphere
double hypersphere_surface_area(int n, double radius) {
    if (n <= 0) return 0;
    
    double numerator = 2 * pow(PI, n/2.0) * pow(radius, n-1);
    double denominator = gamma_function(n/2.0);
    
    return numerator / denominator;
}

// Distance distribution in high-dimensional spaces
double high_dim_distance_pdf(double r, int d, double sigma) {
    // Distribution of distances in d-dimensional Gaussian
    if (r < 0) return 0;
    
    double coeff = 2 * pow(r, d-1) * exp(-r*r/(2*sigma*sigma));
    double denom = pow(2*PI*sigma*sigma, d/2.0) * gamma_function(d/2.0);
    
    return coeff / denom;
}

// Curvature tensor calculation for hypergeometry
void calculate_curvature_tensor(int dimensions, double* metric, double* curvature) {
    // Simplified curvature calculation for demonstration
    // In practice, this would involve Christoffel symbols and Riemann tensor
    
    for (int i = 0; i < dimensions; i++) {
        for (int j = 0; j < dimensions; j++) {
            for (int k = 0; k < dimensions; k++) {
                for (int l = 0; l < dimensions; l++) {
                    int idx = i*dimensions*dimensions*dimensions + 
                              j*dimensions*dimensions + 
                              k*dimensions + l;
                    
                    // Simplified curvature (actual calculation is complex)
                    curvature[idx] = (metric[i*dimensions + k] * metric[j*dimensions + l] -
                                      metric[i*dimensions + l] * metric[j*dimensions + k]) * 0.1;
                }
            }
        }
    }
}

// =============================
// HYPERGEOMETRIC DISTRIBUTIONS
// =============================

// Multivariate hypergeometric distribution PMF
double multivariate_hypergeometric_pmf(int* draws, int* successes, int* population, 
                                       int categories, int total_draws) {
    double log_prob = 0.0;
    int total_population = 0;
    
    // Calculate total population
    for (int i = 0; i < categories; i++) {
        total_population += population[i];
    }
    
    // Log of multinomial coefficient
    for (int i = 0; i < categories; i++) {
        log_prob += lgamma(successes[i] + 1) - lgamma(draws[i] + 1) - lgamma(successes[i] - draws[i] + 1);
    }
    
    log_prob += lgamma(total_population - total_draws + 1) + lgamma(total_draws + 1) - lgamma(total_population + 1);
    
    return exp(log_prob);
}

// Generalized hypergeometric distribution
double generalized_hypergeometric(double a[], int p, double b[], int q, double z) {
    double result = 1.0;
    double term = 1.0;
    
    for (int n = 1; n < 500; n++) {
        double numerator = 1.0;
        double denominator = 1.0;
        
        for (int i = 0; i < p; i++) {
            numerator *= (a[i] + n - 1);
        }
        
        for (int i = 0; i < q; i++) {
            denominator *= (b[i] + n - 1);
        }
        
        term *= (numerator / denominator) * (z / n);
        result += term;
        
        if (fabs(term) < EPSILON) break;
    }
    
    return result;
}

// Volume element in curved space
double volume_element(int dimensions, double* metric, double* coordinates) {
    double det_g = 1.0;
    
    // Simplified determinant calculation
    for (int i = 0; i < dimensions; i++) {
        det_g *= metric[i*dimensions + i] + EPSILON;
    }
    
    return sqrt(fabs(det_g));
}

// =============================
// RANDOM SAMPLING IN HIGH DIMENSIONS
// =============================

// Generate random point on n-sphere (uniform distribution)
void random_point_on_nsphere(int n, double radius, double* point) {
    double norm = 0.0;
    
    // Generate normal distribution coordinates
    for (int i = 0; i < n; i++) {
        double u1 = (double)rand() / RAND_MAX;
        double u2 = (double)rand() / RAND_MAX;
        
        // Box-Muller transform for normal distribution
        point[i] = sqrt(-2 * log(u1)) * cos(2 * PI * u2);
        norm += point[i] * point[i];
    }
    
    norm = sqrt(norm);
    
    // Normalize to sphere surface
    for (int i = 0; i < n; i++) {
        point[i] = radius * point[i] / norm;
    }
}

// Generate random point in n-ball (uniform distribution)
void random_point_in_nball(int n, double radius, double* point) {
    // Generate point on sphere
    random_point_on_nsphere(n, 1.0, point);
    
    // Adjust radius with proper distribution
    double u = pow((double)rand() / RAND_MAX, 1.0/n);
    
    for (int i = 0; i < n; i++) {
        point[i] *= radius * u;
    }
}

// Sample from von Mises-Fisher distribution (directional distribution)
void sample_von_mises_fisher(int n, double* mean_direction, double kappa, double* sample) {
    // Generate sample using rejection sampling
    double beta = (sqrt(4*kappa*kappa + (n-1)*(n-1)) - (n-1)) / (2*kappa);
    double x0 = (1 - beta) / (1 + beta);
    
    double w;
    do {
        double z = beta_rand(0.5*(n-1), 0.5*(n-1));
        w = (1 - (1+beta)*z) / (1 - (1-beta)*z);
    } while (isnan(w) || w < -1 || w > 1);
    
    double t = kappa*w + sqrt(1-w*w);
    
    // Generate orthogonal components
    double* v = malloc((n-1) * sizeof(double));
    double v_norm = 0.0;
    
    for (int i = 0; i < n-1; i++) {
        v[i] = randn();
        v_norm += v[i] * v[i];
    }
    
    v_norm = sqrt(v_norm);
    for (int i = 0; i < n-1; i++) {
        v[i] /= v_norm;
    }
    
    // Construct sample
    sample[0] = w;
    for (int i = 1; i < n; i++) {
        sample[i] = sqrt(1-w*w) * v[i-1];
    }
    
    free(v);
}

// =============================
// DISTANCE AND SIMILARITY METRICS
// =============================

// Mahalanobis distance in high dimensions
double mahalanobis_distance(int n, double* x, double* y, double* inv_covariance) {
    double diff[n];
    double result = 0.0;
    
    for (int i = 0; i < n; i++) {
        diff[i] = x[i] - y[i];
    }
    
    for (int i = 0; i < n; i++) {
        double temp = 0.0;
        for (int j = 0; j < n; j++) {
            temp += diff[j] * inv_covariance[i*n + j];
        }
        result += temp * diff[i];
    }
    
    return sqrt(result);
}

// Cosine similarity in high dimensions
double cosine_similarity(int n, double* a, double* b) {
    double dot = 0.0;
    double norm_a = 0.0;
    double norm_b = 0.0;
    
    for (int i = 0; i < n; i++) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    
    if (norm_a < EPSILON || norm_b < EPSILON) return 0.0;
    
    return dot / (sqrt(norm_a) * sqrt(norm_b));
}

// Jensen-Shannon divergence
double jensen_shannon_divergence(double* p, double* q, int n) {
    double m[n];
    double jsd = 0.0;
    
    for (int i = 0; i < n; i++) {
        m[i] = 0.5 * (p[i] + q[i] + EPSILON);
    }
    
    for (int i = 0; i < n; i++) {
        if (p[i] > EPSILON) {
            jsd += p[i] * log(p[i] / m[i]);
        }
        if (q[i] > EPSILON) {
            jsd += q[i] * log(q[i] / m[i]);
        }
    }
    
    return 0.5 * jsd;
}

// =============================
// ADVANCED VISUALIZATION
// =============================

// Project high-dimensional data to 3D using PCA
void pca_projection_3d(double** data, int samples, int dimensions, double** projection) {
    // Simplified PCA implementation
    // In practice, use eigenvalue decomposition of covariance matrix
    
    // Center the data
    double* mean = calloc(dimensions, sizeof(double));
    
    for (int i = 0; i < samples; i++) {
        for (int j = 0; j < dimensions; j++) {
            mean[j] += data[i][j];
        }
    }
    
    for (int j = 0; j < dimensions; j++) {
        mean[j] /= samples;
    }
    
    // Simple random projection for demonstration
    // Real PCA would compute eigenvectors here
    double projection_matrix[3][MAX_DIMENSIONS];
    
    srand(time(NULL));
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < dimensions; j++) {
            projection_matrix[i][j] = (double)rand() / RAND_MAX - 0.5;
        }
    }
    
    // Project data
    for (int i = 0; i < samples; i++) {
        projection[i][0] = projection[i][1] = projection[i][2] = 0.0;
        
        for (int j = 0; j < dimensions; j++) {
            double centered = data[i][j] - mean[j];
            projection[i][0] += centered * projection_matrix[0][j];
            projection[i][1] += centered * projection_matrix[1][j];
            projection[i][2] += centered * projection_matrix[2][j];
        }
    }
    
    free(mean);
}

// Generate heatmap of distance distribution
void generate_distance_heatmap(int dimensions, double radius, int bins, double* heatmap) {
    for (int i = 0; i < bins; i++) {
        double r = (double)i * radius / bins;
        heatmap[i] = high_dim_distance_pdf(r, dimensions, 1.0);
    }
}

// =============================
// MONTE CARLO INTEGRATION IN HIGH DIMENSIONS
// =============================

// Monte Carlo integration over hypersphere
double monte_carlo_hypersphere(int dimensions, double radius, 
                               double (*func)(double*, int), int iterations) {
    double sum = 0.0;
    double* point = malloc(dimensions * sizeof(double));
    
    for (int i = 0; i < iterations; i++) {
        random_point_in_nball(dimensions, radius, point);
        sum += func(point, dimensions);
    }
    
    free(point);
    
    double volume = hypersphere_volume(dimensions, radius);
    return (sum / iterations) * volume;
}

// Importance sampling in high dimensions
double importance_sampling(int dimensions, 
                          double (*target)(double*, int),
                          double (*proposal)(double*, int),
                          int samples) {
    double sum = 0.0;
    double* point = malloc(dimensions * sizeof(double));
    
    for (int i = 0; i < samples; i++) {
        // Generate sample from proposal (simplified)
        for (int j = 0; j < dimensions; j++) {
            point[j] = (double)rand() / RAND_MAX - 0.5;
        }
        
        double weight = target(point, dimensions) / (proposal(point, dimensions) + EPSILON);
        sum += weight;
    }
    
    free(point);
    return sum / samples;
}

// =============================
// UTILITY FUNCTIONS
// =============================

double randn() {
    // Generate standard normal random variable
    double u1 = (double)rand() / RAND_MAX;
    double u2 = (double)rand() / RAND_MAX;
    return sqrt(-2 * log(u1)) * cos(2 * PI * u2);
}

double beta_rand(double a, double b) {
    // Generate beta random variable
    double x = gamma_rand(a, 1);
    double y = gamma_rand(b, 1);
    return x / (x + y);
}

double gamma_rand(double shape, double scale) {
    // Generate gamma random variable
    if (shape < 1) {
        double u = (double)rand() / RAND_MAX;
        return gamma_rand(1 + shape, scale) * pow(u, 1/shape);
    }
    
    double d = shape - 1.0/3.0;
    double c = 1.0 / sqrt(9 * d);
    
    double v;
    do {
        double x;
        do {
            x = randn();
            v = 1 + c * x;
        } while (v <= 0);
        
        v = v * v * v;
        double u = (double)rand() / RAND_MAX;
        
        if (u < 1 - 0.0331 * (x*x) * (x*x)) {
            return scale * d * v;
        }
    } while (log(u) > 0.5 * x*x + d * (1 - v + log(v)));
    
    return scale * d * v;
}

// =============================
// HYPERGEOMETRY ANALYSIS
// =============================

void analyze_hypergeometry(HyperGeometry* geometry, HyperStatisticalData* stats) {
    printf("\n=== HYPERGEOMETRY ANALYSIS ===\n");
    printf("Dimensions: %d\n", geometry->dimension);
    printf("Theoretical Volume: %.6Le\n", geometry->volume);
    printf("Theoretical Surface Area: %.6Le\n", geometry->surface_area);
    
    // Calculate empirical statistics
    double min_dist = DBL_MAX;
    double max_dist = 0;
    double avg_dist = 0;
    
    int samples = HYPERSPHERE_SAMPLES;
    if (samples > HYPERCUBE_VERTICES) samples = HYPERCUBE_VERTICES;
    
    for (int i = 0; i < samples; i++) {
        // Calculate distance from origin
        double dist = 0;
        for (int d = 0; d < geometry->dimension; d++) {
            dist += geometry->points[i].coordinates[d] * 
                    geometry->points[i].coordinates[d];
        }
        dist = sqrt(dist);
        
        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
        avg_dist += dist;
        
        // Update distribution
        int bin = (int)(dist * 100 / max_dist);
        if (bin >= 0 && bin < 1000) {
            stats->distance_distribution[bin]++;
        }
    }
    
    avg_dist /= samples;
    
    printf("Minimum Distance: %.6f\n", min_dist);
    printf("Maximum Distance: %.6f\n", max_dist);
    printf("Average Distance: %.6f\n", avg_dist);
    printf("Concentration Measure: %.6f\n", avg_dist / sqrt(geometry->dimension));
}

// =============================
// DEMONSTRATION FUNCTIONS
// =============================

// Example target function for integration
double example_function(double* point, int dimensions) {
    double sum = 0.0;
    for (int i = 0; i < dimensions; i++) {
        sum += point[i] * point[i];
    }
    return exp(-sum);
}

// Example proposal distribution
double example_proposal(double* point, int dimensions) {
    double sum = 0.0;
    for (int i = 0; i < dimensions; i++) {
        sum += point[i] * point[i];
    }
    return exp(-sum / 2.0) / pow(2 * PI, dimensions / 2.0);
}

// =============================
// MAIN DEMONSTRATION
// =============================
int main() {
    printf("========================================\n");
    printf("    HYPERGEOMETRY DISTRIBUTION ANALYSIS\n");
    printf("    Advanced Mathematics in C Language\n");
    printf("========================================\n\n");
    
    srand(time(NULL));
    
    // Initialize hypergeometry
    HyperGeometry geometry;
    geometry.dimension = 16;  // 16-dimensional space
    geometry.volume = hypersphere_volume(geometry.dimension, 1.0);
    geometry.surface_area = hypersphere_surface_area(geometry.dimension, 1.0);
    
    // Generate hypercube vertices (simplified)
    printf("Generating hypercube vertices in %d dimensions...\n", geometry.dimension);
    int vertices = 1 << (geometry.dimension < 16 ? geometry.dimension : 16);
    
    for (int i = 0; i < vertices && i < HYPERCUBE_VERTICES; i++) {
        geometry.points[i].dimensions = geometry.dimension;
        
        // Generate binary coordinates
        for (int d = 0; d < geometry.dimension; d++) {
            geometry.points[i].coordinates[d] = ((i >> d) & 1) ? 1.0 : -1.0;
        }
        
        geometry.points[i].weight = 1.0;
        geometry.points[i].curvature = 0.0;
        geometry.points[i].is_vertex = true;
    }
    
    // Initialize statistical data
    HyperStatisticalData stats;
    memset(&stats, 0, sizeof(HyperStatisticalData));
    stats.sample_count = HYPERSPHERE_SAMPLES;
    
    // Perform analysis
    analyze_hypergeometry(&geometry, &stats);
    
    // Demonstrate mathematical functions
    printf("\n=== MATHEMATICAL FUNCTION DEMONSTRATION ===\n");
    
    double a = 2.5, b = 3.5;
    printf("Gamma(%.2f) = %.6f\n", a, gamma_function(a));
    printf("Beta(%.2f, %.2f) = %.6f\n", a, b, beta_function(a, b));
    
    double complex z = 0.5 + 0.3 * I;
    double complex hgf = hypergeometric_2F1(a, b, a + b, z);
    printf("₂F₁(%.2f, %.2f, %.2f, %.2f+%.2fi) = %.6f + %.6fi\n", 
           a, b, a+b, creal(z), cimag(z), creal(hgf), cimag(hgf));
    
    // Demonstrate volume calculations
    printf("\n=== HYPERSPHERE PROPERTIES ===\n");
    for (int d = 1; d <= 10; d++) {
        double vol = hypersphere_volume(d, 1.0);
        double surf = hypersphere_surface_area(d, 1.0);
        printf("Dimension %2d: Volume = %.6e, Surface = %.6e\n", d, vol, surf);
    }
    
    // Monte Carlo integration example
    printf("\n=== MONTE CARLO INTEGRATION ===\n");
    double mc_result = monte_carlo_hypersphere(4, 1.0, example_function, 10000);
    double exact_volume = hypersphere_volume(4, 1.0);
    printf("Monte Carlo result: %.6f\n", mc_result);
    printf("Exact volume: %.6f\n", exact_volume);
    printf("Relative error: %.2f%%\n", fabs(mc_result - exact_volume) / exact_volume * 100);
    
    // Distance distribution analysis
    printf("\n=== DISTANCE DISTRIBUTION ANALYSIS ===\n");
    printf("Dimension: %d\n", geometry.dimension);
    
    // Generate heatmap
    double heatmap[100];
    generate_distance_heatmap(geometry.dimension, 2.0, 100, heatmap);
    
    printf("\nDistance distribution (normalized):\n");
    for (int i = 0; i < 10; i++) {
        printf("Bin %2d: %.4f\n", i, heatmap[i]);
    }
    
    // Demonstrate high-dimensional sampling
    printf("\n=== HIGH-DIMENSIONAL SAMPLING ===\n");
    int sample_dim = 8;
    double sample_point[MAX_DIMENSIONS];
    
    printf("Sampling from %d-dimensional hypersphere:\n", sample_dim);
    random_point_on_nsphere(sample_dim, 1.0, sample_point);
    
    printf("Sample point: [");
    for (int i = 0; i < fmin(sample_dim, 4); i++) {
        printf("%.3f", sample_point[i]);
        if (i < fmin(sample_dim, 4) - 1) printf(", ");
    }
    if (sample_dim > 4) printf(", ...");
    printf("]\n");
    
    // Calculate norm to verify it's on sphere
    double norm = 0.0;
    for (int i = 0; i < sample_dim; i++) {
        norm += sample_point[i] * sample_point[i];
    }
    printf("Norm: %.6f (should be 1.000000)\n", sqrt(norm));
    
    // Curvature analysis
    printf("\n=== CURVATURE ANALYSIS ===\n");
    
    if (geometry.dimension <= 8) {
        int tensor_size = geometry.dimension * geometry.dimension * 
                         geometry.dimension * geometry.dimension;
        double* curvature = malloc(tensor_size * sizeof(double));
        double* metric = malloc(geometry.dimension * geometry.dimension * sizeof(double));
        
        // Initialize simple Euclidean metric
        for (int i = 0; i < geometry.dimension; i++) {
            for (int j = 0; j < geometry.dimension; j++) {
                metric[i * geometry.dimension + j] = (i == j) ? 1.0 : 0.0;
            }
        }
        
        calculate_curvature_tensor(geometry.dimension, metric, curvature);
        
        printf("Curvature tensor (first few components):\n");
        int count = 0;
        for (int i = 0; i < geometry.dimension && count < 5; i++) {
            for (int j = 0; j < geometry.dimension && count < 5; j++) {
                for (int k = 0; k < geometry.dimension && count < 5; k++) {
                    for (int l = 0; l < geometry.dimension && count < 5; l++) {
                        int idx = i*geometry.dimension*geometry.dimension*geometry.dimension + 
                                  j*geometry.dimension*geometry.dimension + 
                                  k*geometry.dimension + l;
                        printf("R[%d][%d][%d][%d] = %.6f\n", i, j, k, l, curvature[idx]);
                        count++;
                    }
                }
            }
        }
        
        free(curvature);
        free(metric);
    }
    
    // Concentration of measure phenomenon
    printf("\n=== CONCENTRATION OF MEASURE ===\n");
    printf("The phenomenon where high-dimensional distributions concentrate:\n");
    
    int test_dims[] = {2, 4, 8, 16, 32};
    for (int i = 0; i < 5; i++) {
        int dim = test_dims[i];
        double expected_radius = sqrt(dim);  // For Gaussian distribution
        
        // Empirical test
        double total_dist = 0.0;
        int trials = 1000;
        
        for (int t = 0; t < trials; t++) {
            double point[dim];
            for (int d = 0; d < dim; d++) {
                point[d] = randn();
            }
            
            double dist = 0.0;
            for (int d = 0; d < dim; d++) {
                dist += point[d] * point[d];
            }
            total_dist += sqrt(dist);
        }
        
        double avg_dist = total_dist / trials;
        printf("Dimension %2d: Expected √d = %.2f, Empirical avg = %.2f, Ratio = %.4f\n",
               dim, expected_radius, avg_dist, avg_dist / expected_radius);
    }
    
    // Information geometry example
    printf("\n=== INFORMATION GEOMETRY ===\n");
    
    // Two probability distributions
    double p[5] = {0.2, 0.2, 0.2, 0.2, 0.2};  // Uniform
    double q[5] = {0.4, 0.3, 0.2, 0.05, 0.05}; // Skewed
    
    double kl_div = 0.0;
    for (int i = 0; i < 5; i++) {
        if (p[i] > EPSILON && q[i] > EPSILON) {
            kl_div += p[i] * log(p[i] / q[i]);
        }
    }
    
    double jsd = jensen_shannon_divergence(p, q, 5);
    
    printf("KL Divergence D(p||q) = %.6f\n", kl_div);
    printf("Jensen-Shannon Divergence = %.6f\n", jsd);
    
    // Generate random hypergeometric samples
    printf("\n=== HYPERGEOMETRIC SAMPLING ===\n");
    
    int population[] = {10, 15, 20, 25};
    int draws[] = {2, 3, 4, 1};
    int successes[] = {5, 10, 8, 12};
    int categories = 4;
    int total_draws = 10;
    
    double pmf = multivariate_hypergeometric_pmf(draws, successes, population, categories, total_draws);
    printf("Multivariate Hypergeometric PMF = %.8f\n", pmf);
    
    // Generate correlation matrix for high dimensions
    printf("\n=== HIGH-DIMENSIONAL CORRELATION ===\n");
    
    int corr_dim = 5;
    double correlation[corr_dim][corr_dim];
    
    for (int i = 0; i < corr_dim; i++) {
        for (int j = 0; j < corr_dim; j++) {
            if (i == j) {
                correlation[i][j] = 1.0;
            } else {
                correlation[i][j] = 0.5 * exp(-0.1 * fabs(i - j));
            }
        }
    }
    
    printf("Correlation matrix (5x5):\n");
    for (int i = 0; i < corr_dim; i++) {
        for (int j = 0; j < corr_dim; j++) {
            printf("%6.3f ", correlation[i][j]);
        }
        printf("\n");
    }
    
    // Final summary
    printf("\n=== HYPERGEOMETRY INSIGHTS ===\n");
    printf("1. In high dimensions, most volume is near the surface\n");
    printf("2. Distance distributions become concentrated\n");
    printf("3. Curvature effects diminish in very high dimensions\n");
    printf("4. Sampling becomes challenging due to curse of dimensionality\n");
    printf("5. Information geometry provides Riemannian structure to probability spaces\n");
    
    printf("\n========================================\n");
    printf("    HYPERGEOMETRY ANALYSIS COMPLETE\n");
    printf("========================================\n");
    
    return 0;
}