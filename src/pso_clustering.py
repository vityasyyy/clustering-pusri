import numpy as np
from metrics import get_wcss


def run_pso(data, k, n_particles=20, max_iters=100, tolerance=1e-4):
    """Particle Swarm Optimization for Clustering"""
    dim = k * data.shape[1]

    # Initialize Swarm
    particles = np.zeros((n_particles, dim))
    for i in range(n_particles):
        # Initialize particles with random data points to start in a valid area
        particles[i] = data[np.random.choice(data.shape[0], k)].flatten()

    velocities = np.random.uniform(-0.1, 0.1, (n_particles, dim))

    # Personal Bests
    pbest_pos = particles.copy()
    pbest_scores = np.array([get_wcss(p, data, k) for p in particles])

    # Global Best
    gbest_idx = np.argmin(pbest_scores)
    gbest_pos = pbest_pos[gbest_idx].copy()
    gbest_score = pbest_scores[gbest_idx]

    history = [gbest_score]

    # PSO Parameters (based on clerc and kennedy constriction factor)
    w = 0.7298  # Inertia
    c1 = 2.05  # Cognitive (Self find)
    c2 = 2.05  # Social (Swarm find)

    prev_score = float("inf")
    for _ in range(max_iters):
        for i in range(n_particles):
            # invoke random factors for exploration
            r1, r2 = np.random.rand(dim), np.random.rand(dim)

            # Update Velocity
            velocities[i] = (
                w * velocities[i]  # keep moving towards wherever inertia
                + c1 * r1 * (pbest_pos[i] - particles[i])  # personal influence
                + c2 * r2 * (gbest_pos - particles[i])  # swarm influence
            )

            # Update Position
            particles[i] += velocities[i]

            # Evaluate loss
            score = get_wcss(particles[i], data, k)

            # Update Personal Best
            if score < pbest_scores[i]:
                pbest_scores[i] = score
                pbest_pos[i] = particles[i].copy()

            # Update Global Best
            if score < gbest_score:
                gbest_score = score
                gbest_pos = particles[i].copy()

        improvement = abs(prev_score - gbest_score)

        if improvement < tolerance:
            print("\n--- converged before max iter---")
            history.append(gbest_score)
            break

        prev_score = gbest_score
        history.append(gbest_score)

    return gbest_pos, history
