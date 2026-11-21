# Mapillary evidence

Public street-level data from [Mapillary](https://www.mapillary.com/), clipped to the simulated city's box.
Contributor and organisation ids have been removed.

| File | Rows | Columns |
|---|---:|---|
| `mly_traffic_signs.parquet` | 53,278 | `id`, `value` (sign class, e.g. `regulatory--no-entry--g1`), `first_seen_at`, `last_seen_at`, `lon`, `lat` |
| `mly_points.parquet` | 498,300 | same columns; point features such as lane-arrow markings and traffic lights |
| `mly_images.parquet` | 2,324,919 | `sequence_id`, `captured_at`, `compass_angle`, `is_pano`, `lon`, `lat` |
| `mly_sign_detail.parquet` | 9,062 | per direction / turn / truck sign: `id`, `value`, `lon`, `lat`, first / last seen, `aligned_direction`, `n_images` |
| `mly_sign_images.parquet` | 165,629 | which photos saw each of those signs: `sign_id`, `image_id`, `image_lon`, `image_lat`, `compass_angle`, `captured_at`, `is_pano`, `sequence_id` |

Things to know:

- Signs are machine detections: there are false positives, several detections of one physical sign, and a few metres
  of position error.
- A detection does not say which road the sign applies to. The compass angle of the photos that saw it does; that is
  what `src/direction.py` uses.

Licence: © Mapillary contributors, [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
