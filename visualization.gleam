// file: visualization.gleam
import hypercube_248
import gleam/io

/// Simple 2D ASCII visualization of projected vertices
pub fn visualize_2d_projection(cube: hypercube_248.Hypercube248) {
  let projection = hypercube_248.random_projection(cube, 2)
  let scaled_projection = scale_projection(projection, 50, 20)
  
  io.println("\n2D Projection of 248D Cube (ASCII Art):")
  io.println("========================================")
  
  draw_ascii_grid(scaled_projection)
}

fn scale_projection(
  points: List(List(Float)),
  width: Int,
  height: Int
) -> List(List(Int)) {
  // Find min and max values
  let xs = points |> list.map(fn(p) { list.at(p, 0) |> result.unwrap(0.0) })
  let ys = points |> list.map(fn(p) { list.at(p, 1) |> result.unwrap(0.0) })
  
  let min_x = list.fold(xs, 1000.0, fn(acc, x) { 
    case x <. acc { 
      True -> x 
      False -> acc 
    } 
  })
  let max_x = list.fold(xs, -1000.0, fn(acc, x) { 
    case x >. acc { 
      True -> x 
      False -> acc 
    } 
  })
  
  let min_y = list.fold(ys, 1000.0, fn(acc, y) { 
    case y <. acc { 
      True -> y 
      False -> acc 
    } 
  })
  let max_y = list.fold(ys, -1000.0, fn(acc, y) { 
    case y >. acc { 
      True -> y 
      False -> acc 
    } 
  })
  
  // Scale to grid
  points
  |> list.map(fn(point) {
    let x = list.at(point, 0) |> result.unwrap(0.0)
    let y = list.at(point, 1) |> result.unwrap(0.0)
    
    let scaled_x = (x -. min_x) /. (max_x -. min_x) 
                   *. float.from_int(width)
                   |> float.round
                   |> int.from_float
                   
    let scaled_y = (y -. min_y) /. (max_y -. min_y)
                   *. float.from_int(height)
                   |> float.round
                   |> int.from_float
    
    [scaled_x, scaled_y]
  })
}

fn draw_ascii_grid(points: List(List(Int))) {
  let grid_width = 60
  let grid_height = 20
  
  // Initialize empty grid
  let grid = 
    iterator.range(0, grid_height - 1)
    |> iterator.map(fn(_) {
      iterator.range(0, grid_width - 1)
      |> iterator.map(fn(_) { " " })
      |> iterator.to_list
    })
    |> iterator.to_list
  
  // Place points on grid
  let filled_grid = 
    points
    |> list.fold(grid, fn(g, point) {
      case point {
        [x, y] -> {
          let row = list.at(g, y) |> result.unwrap([])
          let new_row = list.set(row, x, "•")
          list.set(g, y, new_row)
        }
        _ -> g
      }
    })
  
  // Draw the grid
  filled_grid
  |> list.map(fn(row) {
    row |> list.join("")
  })
  |> list.map(io.println)
}

/// Generate a summary report of the 248D cube
pub fn generate_summary_report() {
  let cube = hypercube_248.generate_248d_cube()
  let wisdom_vectors = hypercube_248.create_wisdom_space(cube)
  
  io.println("\n=== 248D Wisdom Cube Summary ===")
  io.println("Dimensions: 248")
  io.println("Total Vertices: 2²⁴⁸")
  io.println("Total Wisdom Vectors: " <> 
             list.length(wisdom_vectors) |> int.to_string)
  
  // Calculate some statistics
  let avg_wisdom = wisdom_vectors
    |> list.map(fn(v) { v.wisdom_score })
    |> list.fold(0.0, fn(acc, x) { acc +. x })
    |> fn(total) { total /. float.from_int(list.length(wisdom_vectors)) }
  
  io.println("Average Wisdom Score: " <> float.to_string(avg_wisdom, 4))
  
  // Show top wisdom attributes
  io.println("\nTop Wisdom Attributes:")
  wisdom_vectors
  |> list.take(3)  // Show first 3 vectors
  |> list.map(fn(v) {
    "\nVector " <> int.to_string(v.id) <> ":"
    |> io.println
    
    v.attributes
    |> list.take(5)  // Show first 5 attributes
    |> list.map(fn(attr) {
      "  " <> attr.name <> ": " <> float.to_string(attr.value, 3)
    })
    |> list.map(io.println)
  })
}

/// Main function to demonstrate the system
pub fn main() {
  io.println("🌌 248-Dimensional Hypercube Visualization System 🌌")
  io.println("==================================================")
  
  // 1. Generate and analyze the cube
  hypercube_248.analyze_248d_cube()
  |> io.println
  
  // 2. Generate summary report
  generate_summary_report()
  
  // 3. Create a small example for ASCII visualization
  io.println("\n" <> "="^50)
  io.println("Generating example visualization...")
  
  let example_cube = hypercube_248.generate_248d_cube()
  visualize_2d_projection(example_cube)
  
  io.println("\nLegend:")
  io.println("  • Each dot represents a vertex projected from 248D to 2D")
  io.println("  • Random projection preserves some distance relationships")
  io.println("  • Full structure requires 248 dimensions to fully represent")
}