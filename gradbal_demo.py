#!/usr/bin/env python3
"""
Clean Animation for Presentation - 5 Tasks, No Text, Larger Visuals with Labels
"""

from manim import *
import numpy as np

class GradBalAnimation(Scene):
    def construct(self):
        # Create five gradient vectors with more complex conflict patterns
        origin = ORIGIN
        g1 = Arrow(origin, origin + LEFT*2.5 + UP*0.8, color=RED, buff=0, stroke_width=8)       # Conflicts with most
        g2 = Arrow(origin, origin + LEFT*1.0 + DOWN*2.0, color=ORANGE, buff=0, stroke_width=8)  # Conflicts with some
        g3 = Arrow(origin, origin + RIGHT*2.2 + UP*1.8, color=GREEN, buff=0, stroke_width=8)    # Aligned
        g4 = Arrow(origin, origin + RIGHT*1.8 + UP*2.2, color=BLUE, buff=0, stroke_width=8)     # Aligned
        g5 = Arrow(origin, origin + RIGHT*2.0 + UP*0.5, color=PURPLE, buff=0, stroke_width=8)   # Slightly conflicting
        
        # Add labels for gradients
        g1_label = Text("g1", color=RED, font_size=24).next_to(g1.get_end(), LEFT)
        g2_label = Text("g2", color=ORANGE, font_size=24).next_to(g2.get_end(), DOWN)
        g3_label = Text("g3", color=GREEN, font_size=24).next_to(g3.get_end(), UP+RIGHT)
        g4_label = Text("g4", color=BLUE, font_size=24).next_to(g4.get_end(), UP)
        g5_label = Text("g5", color=PURPLE, font_size=24).next_to(g5.get_end(), RIGHT)
        
        gradients = VGroup(g1, g2, g3, g4, g5)
        labels = VGroup(g1_label, g2_label, g3_label, g4_label, g5_label)
        
        # Show gradients one by one with labels
        for grad, label in zip(gradients, labels):
            self.play(Create(grad), Write(label))
            self.wait(0.3)
        self.wait(2)
        
        # Show original weighted combination BEFORE scaling to demonstrate direction preservation
        original_balanced_direction = (
            (LEFT*2.5 + UP*0.8)*0.2 +      # g1: equal weight initially
            (LEFT*1.0 + DOWN*2.0)*0.2 +    # g2: equal weight initially  
            (RIGHT*2.2 + UP*1.8)*0.2 +     # g3: equal weight initially
            (RIGHT*1.8 + UP*2.2)*0.2 +     # g4: equal weight initially
            (RIGHT*2.0 + UP*0.5)*0.2       # g5: equal weight initially
        )
        temp_arrow = Arrow(origin, origin + original_balanced_direction*1.2, 
                          color=GRAY, buff=0, stroke_width=6, stroke_opacity=0.7)
        temp_label = Text("g_avg", color=GRAY, font_size=20).next_to(temp_arrow.get_end(), UP)
        
        self.play(Create(temp_arrow), Write(temp_label))
        self.wait(1)
        
        # GradBal scaling - scale down conflicting gradients differently
        scaled_g1 = Arrow(origin, origin + (LEFT*2.5 + UP*0.8)*0.4, color=RED, buff=0, stroke_width=8)      # Moderate scaling (preserve direction)
        scaled_g2 = Arrow(origin, origin + (LEFT*1.0 + DOWN*2.0)*0.6, color=ORANGE, buff=0, stroke_width=8) # Medium scaling
        scaled_g5 = Arrow(origin, origin + (RIGHT*2.0 + UP*0.5)*0.8, color=PURPLE, buff=0, stroke_width=8)  # Light scaling
        
        # Update labels for scaled gradients
        g1_label_new = Text("g1", color=RED, font_size=24).next_to(scaled_g1.get_end(), LEFT)
        g2_label_new = Text("g2", color=ORANGE, font_size=24).next_to(scaled_g2.get_end(), DOWN)
        g5_label_new = Text("g5", color=PURPLE, font_size=24).next_to(scaled_g5.get_end(), RIGHT)
        
        self.play(
            Transform(g1, scaled_g1),
            Transform(g2, scaled_g2), 
            Transform(g5, scaled_g5),
            Transform(g1_label, g1_label_new),
            Transform(g2_label, g2_label_new),
            Transform(g5_label, g5_label_new),
            run_time=2
        )
        self.wait(1)
        
        # KEY: GradBal final result preserves ALL directions (including conflicts) 
        # with adjusted weights - this is the crucial difference from PCGrad
        gradbal_balanced_direction = (
            (LEFT*2.5 + UP*0.8)*0.12 +     # g1: reduced but STILL PRESENT
            (LEFT*1.0 + DOWN*2.0)*0.15 +   # g2: reduced but STILL PRESENT
            (RIGHT*2.2 + UP*1.8)*0.28 +    # g3: increased weight
            (RIGHT*1.8 + UP*2.2)*0.28 +    # g4: increased weight  
            (RIGHT*2.0 + UP*0.5)*0.17      # g5: slightly reduced
        )
        
        balanced_arrow = Arrow(origin, origin + gradbal_balanced_direction*1.4, 
                              color=WHITE, buff=0, stroke_width=12)
        balanced_label = Text("g_final", color=WHITE, font_size=28).next_to(balanced_arrow.get_end(), UP)
        
        # Remove the temporary average arrow
        self.play(FadeOut(temp_arrow), FadeOut(temp_label))
        
        # Show the final result that maintains ALL gradient directions
        self.play(Create(balanced_arrow), Write(balanced_label))
        
        # Draw dashed lines showing that ALL original gradients contribute to final result
        contribution_lines = VGroup()
        for i, grad in enumerate([g1, g2, g3, g4, g5]):
            line = DashedLine(
                grad.get_end(), 
                balanced_arrow.get_end(), 
                color=grad.get_color(), 
                stroke_width=2, 
                stroke_opacity=0.4
            )
            contribution_lines.add(line)
        
        self.play(Create(contribution_lines))
        self.wait(1)
        
        # Emphasize result
        self.play(balanced_arrow.animate.set_stroke_width(16), run_time=0.5)
        self.play(balanced_arrow.animate.set_stroke_width(12), run_time=0.5)
        self.wait(40)

class PCGradAnimation(Scene):
    def construct(self):
        # Same initial gradients
        origin = ORIGIN
        g1 = Arrow(origin, origin + LEFT*2.5 + UP*0.8, color=RED, buff=0, stroke_width=8)
        g2 = Arrow(origin, origin + LEFT*1.0 + DOWN*2.0, color=ORANGE, buff=0, stroke_width=8)
        g3 = Arrow(origin, origin + RIGHT*2.2 + UP*1.8, color=GREEN, buff=0, stroke_width=8)
        g4 = Arrow(origin, origin + RIGHT*1.8 + UP*2.2, color=BLUE, buff=0, stroke_width=8)
        g5 = Arrow(origin, origin + RIGHT*2.0 + UP*0.5, color=PURPLE, buff=0, stroke_width=8)
        
        # Add labels
        g1_label = Text("g1", color=RED, font_size=24).next_to(g1.get_end(), LEFT)
        g2_label = Text("g2", color=ORANGE, font_size=24).next_to(g2.get_end(), DOWN)
        g3_label = Text("g3", color=GREEN, font_size=24).next_to(g3.get_end(), UP+RIGHT)
        g4_label = Text("g4", color=BLUE, font_size=24).next_to(g4.get_end(), UP)
        g5_label = Text("g5", color=PURPLE, font_size=24).next_to(g5.get_end(), RIGHT)
        
        gradients = VGroup(g1, g2, g3, g4, g5)
        labels = VGroup(g1_label, g2_label, g3_label, g4_label, g5_label)
        
        for grad, label in zip(gradients, labels):
            self.play(Create(grad), Write(label))
            self.wait(0.3)
        self.wait(2)
        
        # Show projection operations with dashed lines
        proj_lines = VGroup()
        proj_line1 = DashedLine(g1.get_end(), origin + UP*0.8, color=RED, stroke_opacity=0.5)
        proj_line2 = DashedLine(g2.get_end(), origin + RIGHT*0.5, color=ORANGE, stroke_opacity=0.5)
        proj_lines.add(proj_line1, proj_line2)
        
        self.play(Create(proj_lines))
        self.wait(1)
        
        # PCGrad projections - more aggressive removal of conflicts
        pcgrad_g1 = Arrow(origin, origin + UP*0.8, color=RED, buff=0, stroke_width=8)           # Heavily projected
        pcgrad_g2 = Arrow(origin, origin + RIGHT*0.5, color=ORANGE, buff=0, stroke_width=8)     # Projected orthogonally
        
        # Update labels for projected gradients
        g1_label_new = Text("g1", color=RED, font_size=24).next_to(pcgrad_g1.get_end(), UP)
        g2_label_new = Text("g2", color=ORANGE, font_size=24).next_to(pcgrad_g2.get_end(), RIGHT)
        
        self.play(
            Transform(g1, pcgrad_g1),
            Transform(g2, pcgrad_g2),
            Transform(g1_label, g1_label_new),
            Transform(g2_label, g2_label_new),
            run_time=2
        )
        self.wait(1)
        
        # PCGrad result - simple average of projected gradients
        pcgrad_result = Arrow(origin, origin + (
            UP*0.8 + RIGHT*0.5 + 
            RIGHT*2.2 + UP*1.8 + 
            RIGHT*1.8 + UP*2.2 + 
            RIGHT*2.0 + UP*0.5
        )/5, color=YELLOW, buff=0, stroke_width=12)
        pcgrad_label = Text("g_final", color=YELLOW, font_size=28).next_to(pcgrad_result.get_end(), UP)
        
        self.play(Create(pcgrad_result), Write(pcgrad_label))
        
        # Emphasize result
        self.play(pcgrad_result.animate.set_stroke_width(16), run_time=0.5)
        self.play(pcgrad_result.animate.set_stroke_width(12), run_time=0.5)
        self.wait(40)

class CAGradAnimation(Scene):
    def construct(self):
        # Create 5 gradients with clear spatial separation to avoid overlaps
        origin = ORIGIN
        
        # Position gradients in a fan pattern to avoid overlaps
        g1 = Arrow(origin, origin + LEFT*3.0 + DOWN*0.5, color=RED, buff=0, stroke_width=8)      # Most opposing
        g2 = Arrow(origin, origin + LEFT*0.5 + UP*2.5, color=ORANGE, buff=0, stroke_width=8)     # Slightly opposing  
        g3 = Arrow(origin, origin + RIGHT*2.0 + UP*2.0, color=GREEN, buff=0, stroke_width=8)     # Well aligned
        g4 = Arrow(origin, origin + RIGHT*2.5 + UP*1.0, color=BLUE, buff=0, stroke_width=8)      # Well aligned
        g5 = Arrow(origin, origin + RIGHT*1.5 + UP*2.5, color=PURPLE, buff=0, stroke_width=8)    # Well aligned
        
        # Add labels
        g1_label = Text("g1", color=RED, font_size=24).next_to(g1.get_end(), LEFT)
        g2_label = Text("g2", color=ORANGE, font_size=24).next_to(g2.get_end(), UP)
        g3_label = Text("g3", color=GREEN, font_size=24).next_to(g3.get_end(), UP+RIGHT)
        g4_label = Text("g4", color=BLUE, font_size=24).next_to(g4.get_end(), RIGHT)
        g5_label = Text("g5", color=PURPLE, font_size=24).next_to(g5.get_end(), UP)
        
        gradients = VGroup(g1, g2, g3, g4, g5)
        labels = VGroup(g1_label, g2_label, g3_label, g4_label, g5_label)
        
        # Show gradients with clear spacing
        for grad, label in zip(gradients, labels):
            self.play(Create(grad), Write(label))
            self.wait(0.4)
        self.wait(2)
        
        # Compute and show average gradient
        avg_direction = (
            LEFT*3.0 + DOWN*0.5 +     # g1
            LEFT*0.5 + UP*2.5 +       # g2  
            RIGHT*2.0 + UP*2.0 +      # g3
            RIGHT*2.5 + UP*1.0 +      # g4
            RIGHT*1.5 + UP*2.5        # g5
        ) / 5
        
        avg_arrow = Arrow(origin, origin + avg_direction*1.5, 
                         color=GRAY, buff=0, stroke_width=8, stroke_opacity=0.9)
        avg_label = Text("g_avg", color=GRAY, font_size=24).next_to(avg_arrow.get_end(), DOWN)
        
        self.play(Create(avg_arrow), Write(avg_label))
        self.wait(2)
        
        # DRAMATICALLY highlight the most conflicting task (g1)
        giant_g1 = Arrow(origin, origin + LEFT*3.0 + DOWN*0.5, 
                         color=RED, buff=0, stroke_width=20)
        
        self.play(
            Transform(g1, giant_g1),
            g2.animate.set_stroke_opacity(0.4),
            g3.animate.set_stroke_opacity(0.4),
            g4.animate.set_stroke_opacity(0.4), 
            g5.animate.set_stroke_opacity(0.4),
            g2_label.animate.set_opacity(0.4),
            g3_label.animate.set_opacity(0.4),
            g4_label.animate.set_opacity(0.4),
            g5_label.animate.set_opacity(0.4),
            avg_arrow.animate.set_stroke_opacity(0.4),
            avg_label.animate.set_opacity(0.4),
            run_time=2
        )
        self.wait(1.5)
        
        # Restore visibility of other elements
        self.play(
            g2.animate.set_stroke_opacity(1.0),
            g3.animate.set_stroke_opacity(1.0),
            g4.animate.set_stroke_opacity(1.0),
            g5.animate.set_stroke_opacity(1.0),
            g2_label.animate.set_opacity(1.0),
            g3_label.animate.set_opacity(1.0),
            g4_label.animate.set_opacity(1.0),
            g5_label.animate.set_opacity(1.0),
            avg_arrow.animate.set_stroke_opacity(0.7),
            avg_label.animate.set_opacity(0.7),
            g1.animate.set_stroke_width(12),
            run_time=1
        )
        
        # Show constraint region around average
        constraint_radius = 1.2
        constraint_circle = Circle(radius=constraint_radius, color=YELLOW, stroke_opacity=0.6)
        constraint_circle.move_to(origin + avg_direction*1.5)
        
        self.play(Create(constraint_circle))
        self.wait(1)
        
        # CAGrad solution
        shift_toward_worst = np.array([-3.0, -0.5, 0])
        cagrad_direction = avg_direction + shift_toward_worst * 0.3
        
        cagrad_arrow = Arrow(origin, origin + cagrad_direction*1.6, 
                            color=TEAL, buff=0, stroke_width=12)
        cagrad_label = Text("g_final", color=TEAL, font_size=28).next_to(cagrad_arrow.get_end(), DOWN)
        
        # Animate the shift from average toward helping the worst task
        shifting_arrow = avg_arrow.copy().set_color(TEAL)
        shifting_label = avg_label.copy().set_color(TEAL)
        
        self.play(
            Transform(shifting_arrow, cagrad_arrow),
            Transform(shifting_label, cagrad_label),
            run_time=2.5
        )
        self.wait(1)
        
        # Draw connection line showing the adjustment toward worst task
        adjustment_line = DashedLine(
            origin + avg_direction*1.5,
            origin + cagrad_direction*1.6,
            color=RED, stroke_width=4, stroke_opacity=0.8
        )
        self.play(Create(adjustment_line))
        self.wait(1)
        
        # Final emphasis on the solution
        self.play(shifting_arrow.animate.set_stroke_width(16), run_time=0.5)
        self.play(shifting_arrow.animate.set_stroke_width(12), run_time=0.5)
        
        # Return g1 to normal size but keep it slightly emphasized
        self.play(g1.animate.set_stroke_width(10), run_time=0.5)
        
        self.wait(40)

class ComparisonAnimation(Scene):
    def construct(self):
        # Show all three results side by side for comparison
        
        # Original gradients (smaller, center)
        origin = ORIGIN
        scale = 0.8
        g1 = Arrow(origin, origin + (LEFT*2.5 + UP*0.8)*scale, color=RED, buff=0, stroke_width=4)
        g2 = Arrow(origin, origin + (LEFT*1.0 + DOWN*2.0)*scale, color=ORANGE, buff=0, stroke_width=4)
        g3 = Arrow(origin, origin + (RIGHT*2.2 + UP*1.8)*scale, color=GREEN, buff=0, stroke_width=4)
        g4 = Arrow(origin, origin + (RIGHT*1.8 + UP*2.2)*scale, color=BLUE, buff=0, stroke_width=4)
        g5 = Arrow(origin, origin + (RIGHT*2.0 + UP*0.5)*scale, color=PURPLE, buff=0, stroke_width=4)
        
        # Add small labels for original gradients
        g1_label = Text("g1", color=RED, font_size=16).next_to(g1.get_end(), LEFT)
        g2_label = Text("g2", color=ORANGE, font_size=16).next_to(g2.get_end(), DOWN)
        g3_label = Text("g3", color=GREEN, font_size=16).next_to(g3.get_end(), UP)
        g4_label = Text("g4", color=BLUE, font_size=16).next_to(g4.get_end(), UP)
        g5_label = Text("g5", color=PURPLE, font_size=16).next_to(g5.get_end(), RIGHT)
        
        original_grads = VGroup(g1, g2, g3, g4, g5)
        original_labels = VGroup(g1_label, g2_label, g3_label, g4_label, g5_label)
        
        self.play(Create(original_grads), Write(original_labels))
        self.wait(2)
        
        # Results positioned around the original
        result_scale = 1.2
        
        # GradBal result (top-left)
        gradbal_pos = origin + LEFT*3 + UP*2
        gradbal_direction = (
            (LEFT*2.5 + UP*0.8)*0.15 +
            (LEFT*1.0 + DOWN*2.0)*0.15 +
            (RIGHT*2.2 + UP*1.8)*0.25 +
            (RIGHT*1.8 + UP*2.2)*0.25 +
            (RIGHT*2.0 + UP*0.5)*0.20
        )
        gradbal_result = Arrow(gradbal_pos, gradbal_pos + gradbal_direction*result_scale, 
                              color=WHITE, buff=0, stroke_width=10)
        gradbal_label = Text("GradBal", color=WHITE, font_size=20).next_to(gradbal_result.get_end(), UP)
        
        # PCGrad result (top-right)
        pcgrad_pos = origin + RIGHT*3 + UP*2
        pcgrad_direction = (
            UP*0.8 + RIGHT*0.5 + 
            RIGHT*2.2 + UP*1.8 + 
            RIGHT*1.8 + UP*2.2 + 
            RIGHT*2.0 + UP*0.5
        ) / 5
        pcgrad_result = Arrow(pcgrad_pos, pcgrad_pos + pcgrad_direction*result_scale, 
                             color=YELLOW, buff=0, stroke_width=10)
        pcgrad_label = Text("PCGrad", color=YELLOW, font_size=20).next_to(pcgrad_result.get_end(), UP)
        
        # CAGrad result (bottom)
        cagrad_pos = origin + DOWN*3
        avg_direction = (LEFT*2.5 + UP*0.8 + LEFT*1.0 + DOWN*2.0 + RIGHT*2.2 + UP*1.8 + RIGHT*1.8 + UP*2.2 + RIGHT*2.0 + UP*0.5) / 5
        cagrad_direction = avg_direction + (LEFT*2.5 + UP*0.8) * 0.2 + (LEFT*1.0 + DOWN*2.0) * 0.15
        cagrad_result = Arrow(cagrad_pos, cagrad_pos + cagrad_direction*result_scale, 
                             color=TEAL, buff=0, stroke_width=10)
        cagrad_label = Text("CAGrad", color=TEAL, font_size=20).next_to(cagrad_result.get_end(), DOWN)
        
        # Show results sequentially
        self.play(Create(gradbal_result), Write(gradbal_label))
        self.wait(1)
        self.play(Create(pcgrad_result), Write(pcgrad_label))
        self.wait(1)
        self.play(Create(cagrad_result), Write(cagrad_label))
        self.wait(1)
        
        # Final emphasis showing clear differences
        all_results = VGroup(gradbal_result, pcgrad_result, cagrad_result)
        self.play(all_results.animate.set_stroke_width(14), run_time=0.5)
        self.play(all_results.animate.set_stroke_width(10), run_time=0.5)
        self.wait(3)
class DirectionalComparisonAnimation(Scene):
    def construct(self):
        """Shows the directional differences between all three methods clearly"""
        
        # Common original gradients for all methods
        origin = ORIGIN
        
        # Original gradients (same as other animations)
        g1_dir = LEFT*2.5 + UP*0.8
        g2_dir = LEFT*1.0 + DOWN*2.0
        g3_dir = RIGHT*2.2 + UP*1.8
        g4_dir = RIGHT*1.8 + UP*2.2
        g5_dir = RIGHT*2.0 + UP*0.5
        
        # Show original gradients briefly
        original_grads = VGroup(
            Arrow(origin, origin + g1_dir*0.6, color=RED, buff=0, stroke_width=4),
            Arrow(origin, origin + g2_dir*0.6, color=ORANGE, buff=0, stroke_width=4),
            Arrow(origin, origin + g3_dir*0.6, color=GREEN, buff=0, stroke_width=4),
            Arrow(origin, origin + g4_dir*0.6, color=BLUE, buff=0, stroke_width=4),
            Arrow(origin, origin + g5_dir*0.6, color=PURPLE, buff=0, stroke_width=4)
        )
        
        self.play(Create(original_grads))
        self.wait(1)
        
        # Calculate final directions for each method
        
        # 1. GradBal: Soft reweighting (preserves all directions)
        gradbal_direction = (
            g1_dir*0.12 + g2_dir*0.15 + g3_dir*0.28 + g4_dir*0.28 + g5_dir*0.17
        )
        
        # 2. PCGrad: Hard projection (eliminates conflicts)
        pcgrad_direction = (
            UP*0.8 + RIGHT*0.5 + g3_dir + g4_dir + g5_dir
        ) / 5
        
        # 3. CAGrad: Constrained optimization (helps worst case)
        avg_direction = (g1_dir + g2_dir + g3_dir + g4_dir + g5_dir) / 5
        cagrad_direction = avg_direction + g1_dir * 0.15  # Shift toward worst case
        
        # Position the three results in a triangular arrangement
        scale = 1.8
        
        # GradBal (top-left)
        gradbal_pos = origin + LEFT*3 + UP*2
        gradbal_arrow = Arrow(gradbal_pos, gradbal_pos + gradbal_direction*scale, 
                             color=WHITE, buff=0, stroke_width=10)
        gradbal_label = Text("GradBal", color=WHITE, font_size=24).next_to(gradbal_arrow.get_end(), UP)
        
        # PCGrad (top-right)  
        pcgrad_pos = origin + RIGHT*3 + UP*2
        pcgrad_arrow = Arrow(pcgrad_pos, pcgrad_pos + pcgrad_direction*scale, 
                            color=YELLOW, buff=0, stroke_width=10)
        pcgrad_label = Text("PCGrad", color=YELLOW, font_size=24).next_to(pcgrad_arrow.get_end(), UP)
        
        # CAGrad (bottom-center)
        cagrad_pos = origin + DOWN*2.5
        cagrad_arrow = Arrow(cagrad_pos, cagrad_pos + cagrad_direction*scale, 
                           color=TEAL, buff=0, stroke_width=10)
        cagrad_label = Text("CAGrad", color=TEAL, font_size=24).next_to(cagrad_arrow.get_end(), DOWN)
        
        # Show results sequentially with emphasis on directional differences
        self.play(Create(gradbal_arrow), Write(gradbal_label))
        self.wait(1)
        
        self.play(Create(pcgrad_arrow), Write(pcgrad_label))  
        self.wait(1)
        
        self.play(Create(cagrad_arrow), Write(cagrad_label))
        self.wait(1)
        
        # Draw angle arcs to show directional differences
        angle_arcs = VGroup()
        
        # Angle between GradBal and PCGrad
        gradbal_angle = np.arctan2(gradbal_direction[1], gradbal_direction[0])
        pcgrad_angle = np.arctan2(pcgrad_direction[1], pcgrad_direction[0])
        
        arc1 = Arc(
            start_angle=min(gradbal_angle, pcgrad_angle),
            angle=abs(gradbal_angle - pcgrad_angle),
            radius=1.0,
            color=GREEN,
            stroke_width=4
        ).move_to(origin + UP*3)
        
        # Angle between CAGrad and average of the other two
        avg_other_angle = (gradbal_angle + pcgrad_angle) / 2
        cagrad_angle = np.arctan2(cagrad_direction[1], cagrad_direction[0])
        
        arc2 = Arc(
            start_angle=min(avg_other_angle, cagrad_angle),
            angle=abs(avg_other_angle - cagrad_angle),
            radius=0.8,
            color=RED,
            stroke_width=4
        ).move_to(origin)
        
        angle_arcs.add(arc1, arc2)
        self.play(Create(angle_arcs))
        self.wait(1)
        
        # Fade out original gradients and emphasize differences
        self.play(FadeOut(original_grads))
        
        # Show magnitude differences with pulsing
        all_arrows = VGroup(gradbal_arrow, pcgrad_arrow, cagrad_arrow)
        
        # Pulse each arrow to show magnitude differences
        for arrow in all_arrows:
            self.play(arrow.animate.set_stroke_width(14), run_time=0.4)
            self.play(arrow.animate.set_stroke_width(10), run_time=0.4)
        
        # Final simultaneous emphasis showing all three distinct directions
        self.play(all_arrows.animate.set_stroke_width(12), run_time=0.5)
        self.play(all_arrows.animate.set_stroke_width(10), run_time=0.5)
        
        self.wait(3)

if __name__ == "__main__":
    print("To render 5-task animations:")
    print("manim gradbal_demo.py GradBalAnimation -pql")
    print("manim gradbal_demo.py PCGradAnimation -pql") 
    print("manim gradbal_demo.py CAGradAnimation -pql")
    print("manim gradbal_demo.py ComparisonAnimation -pql")
    print("manim gradbal_demo.py DirectionalComparisonAnimation -pql")