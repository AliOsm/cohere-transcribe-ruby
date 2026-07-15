# frozen_string_literal: true

require "numo/narray"

module Cohere
  module Transcribe
    module Alignment
      Segment = Data.define(:label, :start, :end)

      # Ruby port of TorchAudio's CPU forced_align kernel. The two-row score
      # matrix remains float32 so its Viterbi choices follow the reference
      # kernel's rounding and tie-breaking behavior, while the back-pointers
      # use one byte per state and frame.
      module CTC
        module_function

        def forced_align(log_probs, targets, blank:)
          frame_count, class_count = matrix_shape(log_probs)
          validate_targets!(targets, blank, class_count)

          target_count = targets.length
          repeat_count = targets.each_cons(2).count { |left, right| left == right }
          minimum_frames = target_count + repeat_count
          if frame_count < minimum_frames
            raise ArgumentError,
                  "targets length is too long for CTC: " \
                  "#{frame_count} frames, #{target_count} targets, #{repeat_count} repeats"
          end

          state_count = (2 * target_count) + 1
          alphas = Numo::SFloat.zeros(2, state_count).fill(-Float::INFINITY)
          back_pointers = "\xFF".b * (frame_count * state_count)

          first_state = frame_count > minimum_frames ? 0 : 1
          end_state = state_count == 1 ? 1 : 2
          (first_state...end_state).each do |state|
            alphas[0, state] = probability(log_probs, 0, label_for(state, targets, blank))
          end

          1.upto(frame_count - 1) do |frame|
            if frame_count - frame <= minimum_frames
              first_state += 1 if first_state.odd? && targets.fetch(first_state / 2) != targets.fetch((first_state / 2) + 1)
              first_state += 1
            end
            if frame <= minimum_frames
              if end_state.even? && end_state < 2 * target_count &&
                 targets.fetch((end_state / 2) - 1) != targets.fetch(end_state / 2)
                end_state += 1
              end
              end_state += 1
            end

            current_row = frame % 2
            previous_row = (frame - 1) % 2
            alphas[current_row, true] = -Float::INFINITY
            loop_start = first_state
            if first_state.zero?
              alphas[current_row, 0] = alphas[previous_row, 0] + probability(log_probs, frame, blank)
              back_pointers.setbyte(frame * state_count, 0)
              loop_start += 1
            end

            (loop_start...end_state).each do |state|
              stay = alphas[previous_row, state]
              advance = alphas[previous_row, state - 1]
              skip = -Float::INFINITY
              if state.odd? && state != 1 && targets.fetch(state / 2) != targets.fetch((state / 2) - 1)
                skip = alphas[previous_row, state - 2]
              end

              score, step = if skip > advance && skip > stay
                              [skip, 2]
                            elsif advance > stay && advance > skip
                              [advance, 1]
                            else
                              [stay, 0]
                            end
              back_pointers.setbyte((frame * state_count) + state, step)
              alphas[current_row, state] =
                score + probability(log_probs, frame, label_for(state, targets, blank))
            end
          end

          final_row = (frame_count - 1) % 2
          state = if alphas[final_row, state_count - 1] > alphas[final_row, state_count - 2]
                    state_count - 1
                  else
                    state_count - 2
                  end
          path = Array.new(frame_count)
          (frame_count - 1).downto(0) do |frame|
            path[frame] = label_for(state, targets, blank)
            break if frame.zero?

            step = back_pointers.getbyte((frame * state_count) + state)
            raise "CTC back-pointer was not initialized" unless step && step <= 2

            state -= step
          end
          path.freeze
        end

        def merge_repeats(path, index_to_token)
          segments = []
          first = 0
          while first < path.length
            following = first + 1
            following += 1 while following < path.length && path.fetch(first) == path.fetch(following)
            label = index_to_token.fetch(path.fetch(first))
            segments << Segment.new(label: label, start: first, end: following - 1)
            first = following
          end
          segments.freeze
        end

        def spans(tokens, segments, blank)
          letter_index = 0
          token_index = 0
          intervals = []
          interval_start = 0

          segments.each_with_index do |segment, segment_index|
            if token_index == tokens.length
              unless segment_index == segments.length - 1 && segment.label == blank
                raise "CTC path continued with a non-blank label after the transcript"
              end

              next
            end

            token_letters = tokens.fetch(token_index).split
            letter = token_letters.fetch(letter_index)
            next if segment.label == blank
            raise "#{segment.label} != #{letter}" unless segment.label == letter

            interval_start = segment_index if letter_index.zero?
            if letter_index == token_letters.length - 1
              letter_index = 0
              token_index += 1
              intervals << [interval_start, segment_index]
              while token_index < tokens.length && tokens.fetch(token_index).empty?
                intervals << [segment_index, segment_index]
                token_index += 1
              end
            else
              letter_index += 1
            end
          end
          raise "CTC path did not consume every transcript token" unless token_index == tokens.length

          intervals.each_with_index.map do |(first, last), interval_index|
            span = segments[first..last]
            if first.positive?
              previous = segments.fetch(first - 1)
              if previous.label == blank
                padded_start = if interval_index.zero?
                                 previous.start
                               else
                                 ((previous.start + previous.end) / 2.0).to_i
                               end
                span = [Segment.new(label: blank, start: padded_start, end: span.first.start), *span]
              end
            end
            if last + 1 < segments.length
              following = segments.fetch(last + 1)
              if following.label == blank
                padded_end = if interval_index == intervals.length - 1
                               following.end
                             else
                               ((following.start + following.end) / 2.0).floor
                             end
                span = [*span, Segment.new(label: blank, start: span.last.end, end: padded_end)]
              end
            end
            span.freeze
          end.freeze
        end

        def matrix_shape(values)
          if values.respond_to?(:ndim)
            raise ArgumentError, "CTC emissions must be a two-dimensional matrix" unless values.ndim == 2

            frames, classes = values.shape
          elsif values.is_a?(Array) && values.all?(Array)
            frames = values.length
            classes = values.first&.length || 0
            raise ArgumentError, "CTC emissions rows must have equal widths" unless values.all? { |row| row.length == classes }
          else
            raise ArgumentError, "CTC emissions must be an indexable matrix"
          end
          raise ArgumentError, "CTC emissions must contain at least one frame" unless frames.positive?
          raise ArgumentError, "CTC emissions must contain at least two classes" unless classes >= 2

          [frames, classes]
        end
        private_class_method :matrix_shape

        def validate_targets!(targets, blank, class_count)
          unless blank.is_a?(Integer) && blank.between?(0, class_count - 1)
            raise ArgumentError, "blank must be within the emissions vocabulary"
          end
          unless targets.is_a?(Array) && !targets.empty? && targets.all?(Integer)
            raise ArgumentError, "CTC targets must be a non-empty integer array"
          end
          raise ArgumentError, "CTC targets must not contain blank index #{blank}" if targets.include?(blank)
          return if targets.all? { |target| target.between?(0, class_count - 1) }

          raise ArgumentError, "CTC targets must be within the emissions vocabulary"
        end
        private_class_method :validate_targets!

        def label_for(state, targets, blank)
          state.even? ? blank : targets.fetch(state / 2)
        end
        private_class_method :label_for

        def probability(values, frame, label)
          values.respond_to?(:ndim) ? values[frame, label] : values.fetch(frame).fetch(label)
        end
        private_class_method :probability
      end
    end
  end
end
