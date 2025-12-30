import javax.swing.*;
import java.awt.*;
import java.awt.geom.Path2D;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class DimensionalWisdomSynthesis extends JPanel {
    
    private static final int CANVAS_WIDTH = 1200;
    private static final int CANVAS_HEIGHT = 800;
    private static final int DIMENSIONS = 248;
    private static final int PROJECTION_DIMS = 2;
    
    private List<double[]> wisdomVectors;
    private List<Color> dimensionColors;
    
    public DimensionalWisdomSynthesis() {
        setPreferredSize(new Dimension(CANVAS_WIDTH, CANVAS_HEIGHT));
        setBackground(Color.BLACK);
        
        wisdomVectors = new ArrayList<>();
        dimensionColors = generateColorSpectrum(DIMENSIONS);
        
        generateWisdomVectors();
    }
    
    private List<Color> generateColorSpectrum(int count) {
        List<Color> colors = new ArrayList<>();
        float hueStep = 1.0f / count;
        
        for (int i = 0; i < count; i++) {
            float hue = i * hueStep;
            Color color = Color.getHSBColor(hue, 0.7f, 1.0f);
            colors.add(color);
        }
        return colors;
    }
    
    private void generateWisdomVectors() {
        Random rand = new Random(42); // Seed for reproducibility
        
        for (int i = 0; i < 100; i++) { // 100 wisdom points
            double[] vector = new double[DIMENSIONS];
            
            // Generate vector with mathematical patterns
            for (int d = 0; d < DIMENSIONS; d++) {
                // Create patterns based on dimension relationships
                double basePattern = Math.sin(d * 0.1) * Math.cos(i * 0.05);
                double wisdomPattern = Math.exp(-0.01 * Math.pow(d - i % 50, 2));
                double noise = rand.nextGaussian() * 0.1;
                
                vector[d] = basePattern + wisdomPattern + noise;
            }
            
            wisdomVectors.add(vector);
        }
    }
    
    private double[] projectTo2D(double[] highDimVector) {
        // Simple random projection (Johnson-Lindenstrauss)
        Random rand = new Random(42);
        double[] projection = new double[PROJECTION_DIMS];
        
        for (int i = 0; i < PROJECTION_DIMS; i++) {
            projection[i] = 0;
            for (int d = 0; d < DIMENSIONS; d++) {
                double weight = rand.nextGaussian() / Math.sqrt(DIMENSIONS);
                projection[i] += highDimVector[d] * weight;
            }
        }
        
        // Scale for visualization
        projection[0] = projection[0] * 200 + CANVAS_WIDTH / 2;
        projection[1] = projection[1] * 200 + CANVAS_HEIGHT / 2;
        
        return projection;
    }
    
    private void drawWisdomStar(Graphics2D g2d, double[] center, int dimensions, Color color) {
        // Draw a star representing multi-dimensional wisdom
        Path2D.Double star = new Path2D.Double();
        double radius = 20 + dimensions * 0.5;
        
        for (int i = 0; i < dimensions; i++) {
            double angle = 2 * Math.PI * i / dimensions;
            double x = center[0] + radius * Math.cos(angle);
            double y = center[1] + radius * Math.sin(angle);
            
            if (i == 0) {
                star.moveTo(x, y);
            } else {
                star.lineTo(x, y);
            }
        }
        star.closePath();
        
        g2d.setColor(new Color(color.getRed(), color.getGreen(), color.getBlue(), 100));
        g2d.fill(star);
        
        g2d.setColor(color);
        g2d.setStroke(new BasicStroke(2));
        g2d.draw(star);
    }
    
    private void drawDimensionConnections(Graphics2D g2d, List<double[]> projections) {
        // Draw connections between dimensions
        g2d.setStroke(new BasicStroke(0.5f));
        
        for (int i = 0; i < projections.size(); i++) {
            for (int j = i + 1; j < projections.size(); j++) {
                double[] p1 = projections.get(i);
                double[] p2 = projections.get(j);
                
                double distance = Math.sqrt(
                    Math.pow(p1[0] - p2[0], 2) + 
                    Math.pow(p1[1] - p2[1], 2)
                );
                
                if (distance < 150) {
                    int alpha = (int)(255 * (1 - distance/150));
                    if (alpha > 30) {
                        Color connectColor = new Color(
                            dimensionColors.get(i).getRed(),
                            dimensionColors.get(i).getGreen(),
                            dimensionColors.get(i).getBlue(),
                            alpha
                        );
                        g2d.setColor(connectColor);
                        g2d.drawLine(
                            (int)p1[0], (int)p1[1],
                            (int)p2[0], (int)p2[1]
                        );
                    }
                }
            }
        }
    }
    
    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        Graphics2D g2d = (Graphics2D) g;
        g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING, 
                            RenderingHints.VALUE_ANTIALIAS_ON);
        
        // Project all wisdom vectors to 2D
        List<double[]> projections = new ArrayList<>();
        for (double[] vector : wisdomVectors) {
            projections.add(projectTo2D(vector));
        }
        
        // Draw connections first
        drawDimensionConnections(g2d, projections);
        
        // Draw wisdom stars
        for (int i = 0; i < projections.size(); i++) {
            drawWisdomStar(g2d, projections.get(i), (i % 24) + 5, dimensionColors.get(i % DIMENSIONS));
        }
        
        // Draw title
        g2d.setFont(new Font("Serif", Font.BOLD, 36));
        g2d.setColor(Color.WHITE);
        String title = "248-Dimensional Wisdom Synthesis";
        FontMetrics fm = g2d.getFontMetrics();
        g2d.drawString(title, 
                      (CANVAS_WIDTH - fm.stringWidth(title)) / 2, 
                      50);
        
        // Draw legend
        drawLegend(g2d);
    }
    
    private void drawLegend(Graphics2D g2d) {
        g2d.setFont(new Font("Monospaced", Font.PLAIN, 12));
        g2d.setColor(Color.WHITE);
        
        String[] legend = {
            "• Each star represents a wisdom vector",
            "• Colors represent different dimensions",
            "• Connections show dimensional relationships",
            "• Star points: dimensional complexity",
            "• Radius: vector magnitude",
            String.format("• Total Dimensions: %d", DIMENSIONS)
        };
        
        for (int i = 0; i < legend.length; i++) {
            g2d.drawString(legend[i], 50, CANVAS_HEIGHT - 150 + i * 20);
        }
    }
    
    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            JFrame frame = new JFrame("248-Dimensional Wisdom Synthesis");
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            frame.add(new DimensionalWisdomSynthesis());
            frame.pack();
            frame.setLocationRelativeTo(null);
            frame.setVisible(true);
        });
    }
}