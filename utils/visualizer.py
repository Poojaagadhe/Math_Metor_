"""Visualization utilities for generating mathematical diagrams and graphs"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, Polygon, Wedge
import sympy as sp
from typing import Dict, Any, List, Tuple, Optional
from utils.logger import setup_logger
import uuid
import re

logger = setup_logger(__name__)

# Create visualizations directory
VIZ_DIR = "./data/visualizations"
os.makedirs(VIZ_DIR, exist_ok=True)

class MathVisualizer:
    """Generate visual explanations for math problems"""
    
    def __init__(self):
        """Initialize visualizer"""
        # Set matplotlib style
        plt.style.use('seaborn-v0_8-darkgrid')
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['font.size'] = 11
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 14
        
    def generate_visualization(self, problem_type: str, problem_data: Dict[str, Any]) -> List[str]:
        """
        Generate appropriate visualization based on problem type
        
        Args:
            problem_type: Type of problem (calculus, algebra, geometry, etc.)
            problem_data: Dictionary with problem-specific data
            
        Returns:
            List of image file paths
        """
        try:
            problem_id = str(uuid.uuid4())[:8]
            viz_path = os.path.join(VIZ_DIR, problem_id)
            os.makedirs(viz_path, exist_ok=True)
            
            images = []
            
            if problem_type == "derivative":
                images = self._plot_derivative(problem_data, viz_path)
            elif problem_type == "integral":
                images = self._plot_integral(problem_data, viz_path)
            elif problem_type == "quadratic":
                images = self._plot_quadratic(problem_data, viz_path)
            elif problem_type == "system_equations":
                images = self._plot_system(problem_data, viz_path)
            elif problem_type == "geometry_circle":
                images = self._draw_circle(problem_data, viz_path)
            elif problem_type == "geometry_triangle":
                images = self._draw_triangle(problem_data, viz_path)
            elif problem_type == "geometry_rectangle":
                images = self._draw_rectangle(problem_data, viz_path)
            else:
                logger.warning(f"No visualization available for problem type: {problem_type}")
                return []
            
            logger.info(f"Generated {len(images)} visualization(s) for {problem_type}")
            return images
            
        except Exception as e:
            logger.error(f"Error generating visualization: {str(e)}")
            return []
    
    def _plot_derivative(self, data: Dict[str, Any], viz_path: str) -> List[str]:
        """Plot function and its derivative"""
        try:
            func_str = data.get('function', 'x**2')
            x_range = data.get('x_range', (-5, 5))
            
            # Parse function using sympy
            x = sp.Symbol('x')
            func = sp.sympify(func_str)
            derivative = sp.diff(func, x)
            
            # Create numerical functions
            func_np = sp.lambdify(x, func, 'numpy')
            deriv_np = sp.lambdify(x, derivative, 'numpy')
            
            # Generate x values
            x_vals = np.linspace(x_range[0], x_range[1], 500)
            
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            # Plot original function
            ax1.plot(x_vals, func_np(x_vals), 'b-', linewidth=2, label=f'f(x) = {sp.latex(func)}')
            ax1.grid(True, alpha=0.3)
            ax1.axhline(y=0, color='k', linewidth=0.5)
            ax1.axvline(x=0, color='k', linewidth=0.5)
            ax1.set_xlabel('x')
            ax1.set_ylabel('f(x)')
            ax1.set_title('Original Function', fontsize=14, fontweight='bold')
            ax1.legend()
            
            # Plot derivative
            ax2.plot(x_vals, deriv_np(x_vals), 'r-', linewidth=2, label=f"f'(x) = {sp.latex(derivative)}")
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=0, color='k', linewidth=0.5)
            ax2.axvline(x=0, color='k', linewidth=0.5)
            ax2.set_xlabel('x')
            ax2.set_ylabel("f'(x)")
            ax2.set_title('Derivative', fontsize=14, fontweight='bold')
            ax2.legend()
            
            plt.tight_layout()
            
            # Save figure
            img_path = os.path.join(viz_path, 'derivative.png')
            plt.savefig(img_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return [img_path]
            
        except Exception as e:
            logger.error(f"Error plotting derivative: {str(e)}")
            return []
    
    def _plot_integral(self, data: Dict[str, Any], viz_path: str) -> List[str]:
        """Plot function with shaded area under curve"""
        try:
            func_str = data.get('function', 'x**2')
            bounds = data.get('bounds', (0, 2))
            x_range = data.get('x_range', (bounds[0] - 1, bounds[1] + 1))
            
            # Parse function
            x = sp.Symbol('x')
            func = sp.sympify(func_str)
            func_np = sp.lambdify(x, func, 'numpy')
            
            # Calculate integral
            integral_result = sp.integrate(func, (x, bounds[0], bounds[1]))
            
            # Generate x values
            x_vals = np.linspace(x_range[0], x_range[1], 500)
            x_fill = np.linspace(bounds[0], bounds[1], 500)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot function
            ax.plot(x_vals, func_np(x_vals), 'b-', linewidth=2, label=f'f(x) = {sp.latex(func)}')
            
            # Shade area under curve
            ax.fill_between(x_fill, 0, func_np(x_fill), alpha=0.3, color='blue', 
                           label=f'Area = {float(integral_result):.3f}')
            
            # Mark bounds
            ax.axvline(x=bounds[0], color='r', linestyle='--', linewidth=1.5, label=f'x = {bounds[0]}')
            ax.axvline(x=bounds[1], color='r', linestyle='--', linewidth=1.5, label=f'x = {bounds[1]}')
            
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='k', linewidth=0.5)
            ax.axvline(x=0, color='k', linewidth=0.5)
            ax.set_xlabel('x')
            ax.set_ylabel('f(x)')
            ax.set_title(f'Definite Integral from {bounds[0]} to {bounds[1]}', 
                        fontsize=14, fontweight='bold')
            ax.legend()
            
            plt.tight_layout()
            
            # Save figure
            img_path = os.path.join(viz_path, 'integral.png')
            plt.savefig(img_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return [img_path]
            
        except Exception as e:
            logger.error(f"Error plotting integral: {str(e)}")
            return []
    
    def _plot_quadratic(self, data: Dict[str, Any], viz_path: str) -> List[str]:
        """Plot quadratic function with roots marked"""
        try:
            # Get coefficients (ax^2 + bx + c)
            a = data.get('a', 1)
            b = data.get('b', -5)
            c = data.get('c', 6)
            roots = data.get('roots', [])
            
            # Create function
            x = sp.Symbol('x')
            func = a*x**2 + b*x + c
            func_np = sp.lambdify(x, func, 'numpy')
            
            # Find vertex
            vertex_x = -b / (2*a)
            vertex_y = float(func.subs(x, vertex_x))
            
            # Generate x values
            x_range = (vertex_x - 5, vertex_x + 5)
            x_vals = np.linspace(x_range[0], x_range[1], 500)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot parabola
            ax.plot(x_vals, func_np(x_vals), 'b-', linewidth=2, 
                   label=f'f(x) = {a}x² + {b}x + {c}')
            
            # Mark vertex
            ax.plot(vertex_x, vertex_y, 'go', markersize=10, label=f'Vertex ({vertex_x:.2f}, {vertex_y:.2f})')
            
            # Mark roots if they exist
            if roots:
                for root in roots:
                    ax.plot(root, 0, 'ro', markersize=10)
                    ax.axvline(x=root, color='r', linestyle='--', alpha=0.3)
                ax.plot([], [], 'ro', markersize=10, label=f'Roots: {", ".join([f"{r:.2f}" for r in roots])}')
            
            # Mark axis of symmetry
            ax.axvline(x=vertex_x, color='g', linestyle='--', alpha=0.3, 
                      label=f'Axis: x = {vertex_x:.2f}')
            
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='k', linewidth=0.5)
            ax.axvline(x=0, color='k', linewidth=0.5)
            ax.set_xlabel('x')
            ax.set_ylabel('f(x)')
            ax.set_title('Quadratic Function', fontsize=14, fontweight='bold')
            ax.legend()
            
            plt.tight_layout()
            
            # Save figure
            img_path = os.path.join(viz_path, 'quadratic.png')
            plt.savefig(img_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return [img_path]
            
        except Exception as e:
            logger.error(f"Error plotting quadratic: {str(e)}")
            return []
    
    def _plot_system(self, data: Dict[str, Any], viz_path: str) -> List[str]:
        """Plot system of linear equations with intersection point"""
        try:
            # Get equation parameters (a1*x + b1*y = c1, a2*x + b2*y = c2)
            eq1 = data.get('equation1', {'a': 3, 'b': 2, 'c': 12})
            eq2 = data.get('equation2', {'a': 1, 'b': -1, 'c': 1})
            solution = data.get('solution', {})
            
            # Create x values
            x_vals = np.linspace(-5, 10, 500)
            
            # Calculate y values for each equation
            # a1*x + b1*y = c1 => y = (c1 - a1*x) / b1
            if eq1['b'] != 0:
                y1_vals = (eq1['c'] - eq1['a'] * x_vals) / eq1['b']
            else:
                # Vertical line
                y1_vals = None
                
            if eq2['b'] != 0:
                y2_vals = (eq2['c'] - eq2['a'] * x_vals) / eq2['b']
            else:
                # Vertical line
                y2_vals = None
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot equations
            if y1_vals is not None:
                ax.plot(x_vals, y1_vals, 'b-', linewidth=2, 
                       label=f'{eq1["a"]}x + {eq1["b"]}y = {eq1["c"]}')
            else:
                ax.axvline(x=eq1['c']/eq1['a'], color='b', linewidth=2,
                          label=f'{eq1["a"]}x = {eq1["c"]}')
                
            if y2_vals is not None:
                ax.plot(x_vals, y2_vals, 'r-', linewidth=2, 
                       label=f'{eq2["a"]}x + {eq2["b"]}y = {eq2["c"]}')
            else:
                ax.axvline(x=eq2['c']/eq2['a'], color='r', linewidth=2,
                          label=f'{eq2["a"]}x = {eq2["c"]}')
            
            # Mark intersection point if solution exists
            if solution:
                sol_x = solution.get('x', 0)
                sol_y = solution.get('y', 0)
                ax.plot(sol_x, sol_y, 'go', markersize=12, zorder=5,
                       label=f'Solution: ({sol_x:.2f}, {sol_y:.2f})')
                
                # Add dotted lines to axes
                ax.plot([sol_x, sol_x], [ax.get_ylim()[0], sol_y], 'g--', alpha=0.3)
                ax.plot([ax.get_xlim()[0], sol_x], [sol_y, sol_y], 'g--', alpha=0.3)
            
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='k', linewidth=0.5)
            ax.axvline(x=0, color='k', linewidth=0.5)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title('System of Linear Equations', fontsize=14, fontweight='bold')
            ax.legend()
            
            # Set reasonable limits
            if solution:
                ax.set_xlim(sol_x - 5, sol_x + 5)
                ax.set_ylim(sol_y - 5, sol_y + 5)
            
            plt.tight_layout()
            
            # Save figure
            img_path = os.path.join(viz_path, 'system.png')
            plt.savefig(img_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return [img_path]
            
        except Exception as e:
            logger.error(f"Error plotting system: {str(e)}")
            return []
    
    def _draw_circle(self, data: Dict[str, Any], viz_path: str) -> List[str]:
        """Draw circle with radius and area labeled"""
        try:
            radius = data.get('radius', 5)
            
            # Calculate properties
            area = np.pi * radius**2
            circumference = 2 * np.pi * radius
            
            # Create figure
            fig, ax = plt.subplots(figsize=(8, 8))
            
            # Draw circle
            circle = Circle((0, 0), radius, fill=False, edgecolor='blue', linewidth=2)
            ax.add_patch(circle)
            
            # Draw radius
            ax.plot([0, radius], [0, 0], 'r-', linewidth=2, label=f'Radius = {radius}')
            ax.plot(0, 0, 'ko', markersize=8)  # Center
            ax.plot(radius, 0, 'ro', markersize=8)
            
            # Add labels
            ax.text(radius/2, 0.3, f'r = {radius}', fontsize=12, ha='center',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Add area and circumference text
            ax.text(0, -radius - 1.5, f'Area = πr² = {area:.2f}', fontsize=12, ha='center',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            ax.text(0, -radius - 2.5, f'Circumference = 2πr = {circumference:.2f}', fontsize=12, ha='center',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
            
            # Set equal aspect ratio and limits
            ax.set_aspect('equal')
            ax.set_xlim(-radius - 2, radius + 2)
            ax.set_ylim(-radius - 3, radius + 2)
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='k', linewidth=0.5)
            ax.axvline(x=0, color='k', linewidth=0.5)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title('Circle', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            
            # Save figure
            img_path = os.path.join(viz_path, 'circle.png')
            plt.savefig(img_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return [img_path]
            
        except Exception as e:
            logger.error(f"Error drawing circle: {str(e)}")
            return []
    
    def _draw_triangle(self, data: Dict[str, Any], viz_path: str) -> List[str]:
        """Draw triangle with sides and angles labeled"""
        try:
            # Get triangle data
            sides = data.get('sides', [3, 4, 5])  # Default right triangle
            triangle_type = data.get('type', 'right')
            
            # For right triangle, use simple coordinates
            if triangle_type == 'right' and len(sides) >= 2:
                a, b = sides[0], sides[1]
                c = np.sqrt(a**2 + b**2) if len(sides) < 3 else sides[2]
                
                # Define vertices
                vertices = np.array([[0, 0], [a, 0], [0, b]])
                
            else:
                # General triangle (use first three sides)
                vertices = np.array([[0, 0], [sides[0], 0], [sides[0]/2, sides[1]]])
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Draw triangle
            triangle = Polygon(vertices, fill=False, edgecolor='blue', linewidth=2)
            ax.add_patch(triangle)
            
            # Draw vertices
            ax.plot(vertices[:, 0], vertices[:, 1], 'ro', markersize=8)
            
            # Label sides
            for i in range(3):
                v1 = vertices[i]
                v2 = vertices[(i + 1) % 3]
                mid = (v1 + v2) / 2
                side_length = np.linalg.norm(v2 - v1)
                ax.text(mid[0], mid[1], f'{side_length:.2f}', fontsize=11,
                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
            
            # Calculate and display area
            area = 0.5 * abs(np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0]))
            ax.text(np.mean(vertices[:, 0]), np.mean(vertices[:, 1]) - 0.5,
                   f'Area = {area:.2f}', fontsize=12, ha='center',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            # Set equal aspect and limits
            ax.set_aspect('equal')
            margin = 1
            ax.set_xlim(vertices[:, 0].min() - margin, vertices[:, 0].max() + margin)
            ax.set_ylim(vertices[:, 1].min() - margin, vertices[:, 1].max() + margin)
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title(f'{triangle_type.capitalize()} Triangle', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            
            # Save figure
            img_path = os.path.join(viz_path, 'triangle.png')
            plt.savefig(img_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return [img_path]
            
        except Exception as e:
            logger.error(f"Error drawing triangle: {str(e)}")
            return []
    
    def _draw_rectangle(self, data: Dict[str, Any], viz_path: str) -> List[str]:
        """Draw rectangle with dimensions labeled"""
        try:
            length = data.get('length', 8)
            width = data.get('width', 5)
            
            # Calculate properties
            area = length * width
            perimeter = 2 * (length + width)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Draw rectangle
            rect = patches.Rectangle((0, 0), length, width, fill=False, 
                                    edgecolor='blue', linewidth=2)
            ax.add_patch(rect)
            
            # Draw vertices
            vertices = [[0, 0], [length, 0], [length, width], [0, width]]
            for v in vertices:
                ax.plot(v[0], v[1], 'ro', markersize=8)
            
            # Label dimensions
            ax.text(length/2, -0.5, f'Length = {length}', fontsize=12, ha='center',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
            ax.text(-0.5, width/2, f'Width = {width}', fontsize=12, ha='right', va='center',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
            
            # Add area and perimeter
            ax.text(length/2, width/2, f'Area = {area}', fontsize=13, ha='center',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            ax.text(length/2, width + 1, f'Perimeter = {perimeter}', fontsize=12, ha='center',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
            
            # Set equal aspect and limits
            ax.set_aspect('equal')
            ax.set_xlim(-1, length + 1)
            ax.set_ylim(-1, width + 2)
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title('Rectangle', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            
            # Save figure
            img_path = os.path.join(viz_path, 'rectangle.png')
            plt.savefig(img_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return [img_path]
            
        except Exception as e:
            logger.error(f"Error drawing rectangle: {str(e)}")
            return []
