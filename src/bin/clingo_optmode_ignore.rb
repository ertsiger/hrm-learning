#!/usr/bin/env ruby

args = []

ARGV.each do |arg|
    unless arg.match "--opt-mode"
        args << arg
    end
end

puts `clingo #{args.join(" ")} --opt-mode=ignore --heuristic=domain --dom-mod=5,16`
